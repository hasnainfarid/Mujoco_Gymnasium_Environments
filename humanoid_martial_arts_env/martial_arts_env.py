"""
Humanoid Martial Arts Environment - A MuJoCo-based martial arts training simulation

This environment simulates a humanoid robot learning various martial arts techniques
including strikes, kicks, blocks, and combinations with physics-based movements.

Author: Hasnain Fareed
Year: 2025
License: MIT
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, Dict, Any
import math


class HumanoidMartialArtsEnv(gym.Env):
    """
    A MuJoCo-based humanoid martial arts training environment.
    
    The robot must learn to perform various martial arts techniques including:
    - Basic strikes (punches, palm strikes)
    - Kicks (front kick, side kick, roundhouse)
    - Defensive moves (blocks, dodges)
    - Combinations and forms (kata)
    - Balance and stance transitions
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 60
    }
    
    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        """Initialize the humanoid martial arts environment."""
        super().__init__()
        
        # Environment parameters
        self.dt = 0.01667  # 60 Hz simulation for smooth martial arts movements
        self.max_episode_steps = 6000  # 100 seconds at 60 Hz
        self.current_step = 0
        
        # Training area dimensions (meters)
        self.dojo_length = 12.0
        self.dojo_width = 12.0
        self.mat_thickness = 0.1
        
        # Robot properties
        self.robot_height = 1.75  # Standard humanoid height
        self.num_joints = 27  # Extended for martial arts flexibility
        
        # Technique categories
        self.techniques = {
            'punch': {'reward': 500, 'energy': 0.3},
            'kick': {'reward': 800, 'energy': 0.5},
            'block': {'reward': 300, 'energy': 0.2},
            'dodge': {'reward': 400, 'energy': 0.25},
            'stance': {'reward': 200, 'energy': 0.1},
            'combo': {'reward': 1500, 'energy': 0.8}
        }
        
        # Training dummy properties
        self.dummy_positions = [
            np.array([2.0, 0.0, 1.0]),
            np.array([-2.0, 0.0, 1.0]),
            np.array([0.0, 2.0, 1.0])
        ]
        self.active_dummy_idx = 0
        
        # Reward parameters
        self.technique_reward_scale = 1.0
        self.balance_reward = 100.0
        self.form_accuracy_reward = 300.0
        self.power_generation_reward = 200.0
        self.speed_bonus = 150.0
        self.energy_efficiency_bonus = 100.0
        self.fall_penalty = -500.0
        self.collision_penalty = -200.0
        
        # Performance tracking
        self.combo_chain = []
        self.max_combo_length = 5
        self.stance_stability_time = 0.0
        self.technique_accuracy = 0.0
        
        # Environmental factors
        self.wind_resistance = 0.0
        self.floor_friction = 0.9
        
        # Set render mode
        self.render_mode = render_mode
        
        # Load and combine XML models
        self._load_xml_models()
        
        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)
        
        # Update num_joints based on actual model
        self.num_joints = self.model.nu  # Number of actuators
        
        # Get important indices
        self._get_model_indices()
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Initialize viewer for rendering
        self.viewer = None
        if self.render_mode == "human":
            self._init_viewer()
        
        # Episode tracking
        self.episode_stats = {
            'techniques_performed': 0,
            'successful_combos': 0,
            'balance_maintained': 0,
            'max_power_generated': 0.0,
            'total_distance_moved': 0.0,
            'falls': 0
        }
        
        # Initialize RNG
        self._np_random = None
        self.seed()
        
    def _load_xml_models(self):
        """Load and combine XML model files."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(base_dir, 'assets')
        
        # Create assets directory if it doesn't exist
        os.makedirs(assets_dir, exist_ok=True)
        
        # Generate XML files if they don't exist
        self._generate_xml_files(assets_dir)
        
        # Load main model
        main_xml_path = os.path.join(assets_dir, 'martial_arts_scene.xml')
        with open(main_xml_path, 'r') as f:
            self.xml_string = f.read()
            
    def _generate_xml_files(self, assets_dir):
        """Generate XML model files for the martial arts environment."""
        
        # Main scene XML
        scene_xml = """<?xml version="1.0"?>
<mujoco model="humanoid_martial_arts">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="0.01" damping="0.5" limited="true"/>
        <geom contype="1" conaffinity="1" condim="3" friction="0.9 0.05 0.05" margin="0.01"/>
        <motor ctrlrange="-1 1" ctrllimited="true"/>
    </default>
    
    <option timestep="0.01667" gravity="0 0 -9.81" iterations="50" tolerance="1e-10" solver="Newton" jacobian="dense" cone="pyramidal"/>
    
    <visual>
        <quality shadowsize="2048"/>
        <headlight ambient="0.3 0.3 0.3" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    </visual>
    
    <size njmax="1000" nconmax="500" nstack="600000"/>
    
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
        <texture name="mat" type="2d" builtin="flat" rgb1="0.8 0.6 0.4" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="10 10" specular="0.3" shininess="0.3" reflectance="0.1"/>
        <material name="mat" texture="mat" texrepeat="5 5" specular="0.3" shininess="0.3" reflectance="0.2"/>
    </asset>
    
    <worldbody>
        <!-- Dojo floor -->
        <geom name="floor" type="plane" size="6 6 0.1" material="grid" contype="1" conaffinity="1"/>
        <geom name="training_mat" type="box" pos="0 0 0.05" size="4 4 0.05" material="mat" contype="1" conaffinity="1"/>
        
        <!-- Walls for spatial reference -->
        <geom name="wall_north" type="box" pos="0 6 2" size="6 0.1 2" rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
        <geom name="wall_south" type="box" pos="0 -6 2" size="6 0.1 2" rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
        <geom name="wall_east" type="box" pos="6 0 2" size="0.1 6 2" rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
        <geom name="wall_west" type="box" pos="-6 0 2" size="0.1 6 2" rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>
        
        <!-- Training dummies -->
        <body name="dummy1" pos="2 0 0">
            <joint name="dummy1_base" type="free"/>
            <geom name="dummy1_torso" type="cylinder" size="0.2 0.5" pos="0 0 1" rgba="0.7 0.3 0.3 1" mass="20"/>
            <geom name="dummy1_head" type="sphere" size="0.15" pos="0 0 1.65" rgba="0.7 0.3 0.3 1" mass="5"/>
        </body>
        
        <body name="dummy2" pos="-2 0 0">
            <joint name="dummy2_base" type="free"/>
            <geom name="dummy2_torso" type="cylinder" size="0.2 0.5" pos="0 0 1" rgba="0.3 0.3 0.7 1" mass="20"/>
            <geom name="dummy2_head" type="sphere" size="0.15" pos="0 0 1.65" rgba="0.3 0.3 0.7 1" mass="5"/>
        </body>
        
        <!-- Breaking boards for technique testing -->
        <body name="board1" pos="0 2 1.2">
            <joint name="board1_joint" type="hinge" axis="1 0 0" range="-45 45" damping="0.1"/>
            <geom name="board1_plank" type="box" size="0.3 0.02 0.3" rgba="0.6 0.4 0.2 1" mass="0.5"/>
        </body>
        
        <!-- Humanoid martial artist -->
        <body name="torso" pos="0 0 1.4">
            <joint name="torso_joint" type="free"/>
            <geom name="torso_geom" type="capsule" size="0.15 0.3" rgba="0.8 0.6 0.4 1" mass="10"/>
            
            <!-- Head -->
            <body name="head" pos="0 0 0.5">
                <joint name="neck_pitch" type="hinge" axis="0 1 0" range="-45 45"/>
                <joint name="neck_yaw" type="hinge" axis="0 0 1" range="-90 90"/>
                <geom name="head_geom" type="sphere" size="0.12" rgba="0.8 0.6 0.4 1" mass="3"/>
            </body>
            
            <!-- Right arm -->
            <body name="right_shoulder" pos="0.25 0 0.3">
                <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0" range="-180 90"/>
                <joint name="right_shoulder_roll" type="hinge" axis="1 0 0" range="-90 180"/>
                <joint name="right_shoulder_yaw" type="hinge" axis="0 0 1" range="-90 90"/>
                <geom name="right_upper_arm" type="capsule" fromto="0 0 0 0.3 0 -0.3" size="0.05" rgba="0.8 0.6 0.4 1" mass="2"/>
                
                <body name="right_elbow" pos="0.3 0 -0.3">
                    <joint name="right_elbow_pitch" type="hinge" axis="0 1 0" range="0 150"/>
                    <geom name="right_forearm" type="capsule" fromto="0 0 0 0.25 0 -0.25" size="0.04" rgba="0.8 0.6 0.4 1" mass="1.5"/>
                    
                    <body name="right_hand" pos="0.25 0 -0.25">
                        <joint name="right_wrist_pitch" type="hinge" axis="0 1 0" range="-90 90"/>
                        <joint name="right_wrist_roll" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="right_hand_geom" type="box" size="0.04 0.08 0.02" rgba="0.8 0.6 0.4 1" mass="0.5"/>
                        <site name="right_hand" pos="0 0 0" size="0.01"/>
                    </body>
                </body>
            </body>
            
            <!-- Left arm (mirror of right) -->
            <body name="left_shoulder" pos="-0.25 0 0.3">
                <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0" range="-180 90"/>
                <joint name="left_shoulder_roll" type="hinge" axis="1 0 0" range="-180 90"/>
                <joint name="left_shoulder_yaw" type="hinge" axis="0 0 1" range="-90 90"/>
                <geom name="left_upper_arm" type="capsule" fromto="0 0 0 -0.3 0 -0.3" size="0.05" rgba="0.8 0.6 0.4 1" mass="2"/>
                
                <body name="left_elbow" pos="-0.3 0 -0.3">
                    <joint name="left_elbow_pitch" type="hinge" axis="0 1 0" range="0 150"/>
                    <geom name="left_forearm" type="capsule" fromto="0 0 0 -0.25 0 -0.25" size="0.04" rgba="0.8 0.6 0.4 1" mass="1.5"/>
                    
                    <body name="left_hand" pos="-0.25 0 -0.25">
                        <joint name="left_wrist_pitch" type="hinge" axis="0 1 0" range="-90 90"/>
                        <joint name="left_wrist_roll" type="hinge" axis="1 0 0" range="-45 45"/>
                        <geom name="left_hand_geom" type="box" size="0.04 0.08 0.02" rgba="0.8 0.6 0.4 1" mass="0.5"/>
                        <site name="left_hand" pos="0 0 0" size="0.01"/>
                    </body>
                </body>
            </body>
            
            <!-- Pelvis and legs -->
            <body name="pelvis" pos="0 0 -0.4">
                <joint name="waist_pitch" type="hinge" axis="0 1 0" range="-45 45"/>
                <joint name="waist_yaw" type="hinge" axis="0 0 1" range="-90 90"/>
                <geom name="pelvis_geom" type="box" size="0.15 0.1 0.08" rgba="0.8 0.6 0.4 1" mass="5"/>
                
                <!-- Right leg -->
                <body name="right_hip" pos="0.1 0 -0.1">
                    <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-120 45"/>
                    <joint name="right_hip_roll" type="hinge" axis="1 0 0" range="-45 45"/>
                    <joint name="right_hip_yaw" type="hinge" axis="0 0 1" range="-45 45"/>
                    <geom name="right_thigh" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.06" rgba="0.8 0.6 0.4 1" mass="4"/>
                    
                    <body name="right_knee" pos="0 0 -0.4">
                        <joint name="right_knee_pitch" type="hinge" axis="0 1 0" range="0 150"/>
                        <geom name="right_shin" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" rgba="0.8 0.6 0.4 1" mass="3"/>
                        
                        <body name="right_ankle" pos="0 0 -0.35">
                            <joint name="right_ankle_pitch" type="hinge" axis="0 1 0" range="-45 45"/>
                            <joint name="right_ankle_roll" type="hinge" axis="1 0 0" range="-30 30"/>
                            <geom name="right_foot" type="box" size="0.08 0.15 0.03" pos="0 0.05 -0.03" rgba="0.5 0.3 0.2 1" mass="1"/>
                            <site name="right_foot" pos="0 0 0" size="0.01"/>
                        </body>
                    </body>
                </body>
                
                <!-- Left leg (mirror of right) -->
                <body name="left_hip" pos="-0.1 0 -0.1">
                    <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-120 45"/>
                    <joint name="left_hip_roll" type="hinge" axis="1 0 0" range="-45 45"/>
                    <joint name="left_hip_yaw" type="hinge" axis="0 0 1" range="-45 45"/>
                    <geom name="left_thigh" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.06" rgba="0.8 0.6 0.4 1" mass="4"/>
                    
                    <body name="left_knee" pos="0 0 -0.4">
                        <joint name="left_knee_pitch" type="hinge" axis="0 1 0" range="0 150"/>
                        <geom name="left_shin" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" rgba="0.8 0.6 0.4 1" mass="3"/>
                        
                        <body name="left_ankle" pos="0 0 -0.35">
                            <joint name="left_ankle_pitch" type="hinge" axis="0 1 0" range="-45 45"/>
                            <joint name="left_ankle_roll" type="hinge" axis="1 0 0" range="-30 30"/>
                            <geom name="left_foot" type="box" size="0.08 0.15 0.03" pos="0 0.05 -0.03" rgba="0.5 0.3 0.2 1" mass="1"/>
                            <site name="left_foot" pos="0 0 0" size="0.01"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- Camera for different viewing angles -->
        <camera name="side_view" pos="5 0 2" xyaxes="0 -1 0 0 0 1" fovy="60"/>
        <camera name="front_view" pos="0 -5 2" xyaxes="1 0 0 0 0 1" fovy="60"/>
        <camera name="top_view" pos="0 0 8" xyaxes="1 0 0 0 1 0" fovy="60"/>
    </worldbody>
    
    <actuator>
        <!-- Head actuators -->
        <motor name="neck_pitch_motor" joint="neck_pitch" ctrlrange="-50 50"/>
        <motor name="neck_yaw_motor" joint="neck_yaw" ctrlrange="-50 50"/>
        
        <!-- Right arm actuators -->
        <motor name="right_shoulder_pitch_motor" joint="right_shoulder_pitch" ctrlrange="-100 100"/>
        <motor name="right_shoulder_roll_motor" joint="right_shoulder_roll" ctrlrange="-100 100"/>
        <motor name="right_shoulder_yaw_motor" joint="right_shoulder_yaw" ctrlrange="-100 100"/>
        <motor name="right_elbow_motor" joint="right_elbow_pitch" ctrlrange="-80 80"/>
        <motor name="right_wrist_pitch_motor" joint="right_wrist_pitch" ctrlrange="-40 40"/>
        <motor name="right_wrist_roll_motor" joint="right_wrist_roll" ctrlrange="-40 40"/>
        
        <!-- Left arm actuators -->
        <motor name="left_shoulder_pitch_motor" joint="left_shoulder_pitch" ctrlrange="-100 100"/>
        <motor name="left_shoulder_roll_motor" joint="left_shoulder_roll" ctrlrange="-100 100"/>
        <motor name="left_shoulder_yaw_motor" joint="left_shoulder_yaw" ctrlrange="-100 100"/>
        <motor name="left_elbow_motor" joint="left_elbow_pitch" ctrlrange="-80 80"/>
        <motor name="left_wrist_pitch_motor" joint="left_wrist_pitch" ctrlrange="-40 40"/>
        <motor name="left_wrist_roll_motor" joint="left_wrist_roll" ctrlrange="-40 40"/>
        
        <!-- Waist actuators -->
        <motor name="waist_pitch_motor" joint="waist_pitch" ctrlrange="-50 50"/>
        <motor name="waist_yaw_motor" joint="waist_yaw" ctrlrange="-50 50"/>
        
        <!-- Right leg actuators -->
        <motor name="right_hip_pitch_motor" joint="right_hip_pitch" ctrlrange="-150 150"/>
        <motor name="right_hip_roll_motor" joint="right_hip_roll" ctrlrange="-100 100"/>
        <motor name="right_hip_yaw_motor" joint="right_hip_yaw" ctrlrange="-100 100"/>
        <motor name="right_knee_motor" joint="right_knee_pitch" ctrlrange="-120 120"/>
        <motor name="right_ankle_pitch_motor" joint="right_ankle_pitch" ctrlrange="-60 60"/>
        <motor name="right_ankle_roll_motor" joint="right_ankle_roll" ctrlrange="-60 60"/>
        
        <!-- Left leg actuators -->
        <motor name="left_hip_pitch_motor" joint="left_hip_pitch" ctrlrange="-150 150"/>
        <motor name="left_hip_roll_motor" joint="left_hip_roll" ctrlrange="-100 100"/>
        <motor name="left_hip_yaw_motor" joint="left_hip_yaw" ctrlrange="-100 100"/>
        <motor name="left_knee_motor" joint="left_knee_pitch" ctrlrange="-120 120"/>
        <motor name="left_ankle_pitch_motor" joint="left_ankle_pitch" ctrlrange="-60 60"/>
        <motor name="left_ankle_roll_motor" joint="left_ankle_roll" ctrlrange="-60 60"/>
    </actuator>
    
    <sensor>
        <!-- Force sensors -->
        <force name="right_hand_force" site="right_hand"/>
        <force name="left_hand_force" site="left_hand"/>
        <force name="right_foot_force" site="right_foot"/>
        <force name="left_foot_force" site="left_foot"/>
        
        <!-- Position sensors -->
        <framepos name="torso_pos" objtype="body" objname="torso"/>
        <framequat name="torso_quat" objtype="body" objname="torso"/>
        
        <!-- Velocity sensors -->
        <framelinvel name="torso_linvel" objtype="body" objname="torso"/>
        <frameangvel name="torso_angvel" objtype="body" objname="torso"/>
        
        <!-- Joint sensors -->
        <jointpos name="neck_pos" joint="neck_pitch"/>
        <jointvel name="neck_vel" joint="neck_pitch"/>
    </sensor>
</mujoco>
        """
        
        # Write XML file
        with open(os.path.join(assets_dir, 'martial_arts_scene.xml'), 'w') as f:
            f.write(scene_xml)
            
    def _get_model_indices(self):
        """Get indices for important model elements."""
        # Body indices
        self.torso_idx = self.model.body('torso').id
        self.head_idx = self.model.body('head').id
        self.right_hand_idx = self.model.body('right_hand').id
        self.left_hand_idx = self.model.body('left_hand').id
        self.right_foot_idx = self.model.body('right_ankle').id
        self.left_foot_idx = self.model.body('left_ankle').id
        
        # Dummy indices
        self.dummy1_idx = self.model.body('dummy1').id
        self.dummy2_idx = self.model.body('dummy2').id
        
    def _define_spaces(self):
        """Define action and observation spaces."""
        # Action space: joint torques for all actuators
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # Observation space: comprehensive state information
        obs_dim = (
            3 +  # torso position
            4 +  # torso quaternion
            3 +  # torso linear velocity
            3 +  # torso angular velocity
            self.num_joints +  # joint positions
            self.num_joints +  # joint velocities
            3 * 3 +  # dummy positions (3 dummies)
            4 +  # force sensors (2 hands, 2 feet)
            1 +  # technique accuracy
            1 +  # combo chain length
            1    # stance stability time
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
    def _init_viewer(self):
        """Initialize the MuJoCo viewer for rendering."""
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 8.0
        self.viewer.cam.elevation = -20.0
        self.viewer.cam.azimuth = 45.0
        
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
            
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial humanoid pose (fighting stance)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Set torso position (standing)
        qpos[0:3] = [0, 0, 1.4]  # x, y, z position
        qpos[3:7] = [1, 0, 0, 0]  # quaternion (upright)
        
        # Set initial fighting stance
        if self._np_random is not None:
            # Slight randomization in starting position
            qpos[0] += self._np_random.uniform(-0.5, 0.5)
            qpos[1] += self._np_random.uniform(-0.5, 0.5)
            
        # Apply initial state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        
        # Reset tracking variables
        self.current_step = 0
        self.combo_chain = []
        self.stance_stability_time = 0.0
        self.technique_accuracy = 0.0
        self.active_dummy_idx = 0
        
        # Reset episode statistics
        for key in self.episode_stats:
            self.episode_stats[key] = 0
            
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
        
    def step(self, action):
        """Execute one step in the environment."""
        # Clip and scale actions
        action = np.clip(action, -1.0, 1.0)
        
        # Apply actions to actuators
        self.data.ctrl[:] = action * self.model.actuator_ctrlrange[:, 1]
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Update step counter
        self.current_step += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # Get info
        info = self._get_info()
        
        # Update episode statistics
        self._update_statistics()
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self):
        """Get current observation from the environment."""
        obs = []
        
        # Torso position and orientation
        torso_pos = self.data.xpos[self.torso_idx]
        torso_quat = self.data.xquat[self.torso_idx]
        obs.extend(torso_pos)
        obs.extend(torso_quat)
        
        # Torso velocities
        torso_linvel = self.data.cvel[self.torso_idx][:3]
        torso_angvel = self.data.cvel[self.torso_idx][3:]
        obs.extend(torso_linvel)
        obs.extend(torso_angvel)
        
        # Joint positions and velocities
        obs.extend(self.data.qpos[7:])  # Skip free joint
        obs.extend(self.data.qvel[6:])  # Skip free joint velocities
        
        # Dummy positions
        for dummy_idx in [self.dummy1_idx, self.dummy2_idx]:
            dummy_pos = self.data.xpos[dummy_idx]
            obs.extend(dummy_pos)
        # Add a third dummy position (static)
        obs.extend([0.0, -2.0, 1.0])
        
        # Force sensor readings (simplified)
        obs.extend([0.0, 0.0, 0.0, 0.0])  # Placeholder for force sensors
        
        # Technique metrics
        obs.append(self.technique_accuracy)
        obs.append(len(self.combo_chain))
        obs.append(self.stance_stability_time)
        
        return np.array(obs, dtype=np.float32)
        
    def _calculate_reward(self, action):
        """Calculate reward based on current state and action."""
        reward = 0.0
        
        # Balance reward (staying upright)
        torso_height = self.data.xpos[self.torso_idx][2]
        upright_reward = self.balance_reward * min(1.0, torso_height / self.robot_height)
        reward += upright_reward
        
        # Technique detection (simplified)
        # Check for punch motion
        right_hand_vel = np.linalg.norm(self.data.cvel[self.right_hand_idx][:3])
        left_hand_vel = np.linalg.norm(self.data.cvel[self.left_hand_idx][:3])
        
        if right_hand_vel > 2.0 or left_hand_vel > 2.0:
            reward += self.techniques['punch']['reward']
            self.episode_stats['techniques_performed'] += 1
            
        # Check for kick motion
        right_foot_vel = np.linalg.norm(self.data.cvel[self.right_foot_idx][:3])
        left_foot_vel = np.linalg.norm(self.data.cvel[self.left_foot_idx][:3])
        
        if right_foot_vel > 3.0 or left_foot_vel > 3.0:
            reward += self.techniques['kick']['reward']
            self.episode_stats['techniques_performed'] += 1
            
        # Stance stability bonus
        torso_angular_vel = np.linalg.norm(self.data.cvel[self.torso_idx][3:])
        if torso_angular_vel < 0.5:
            self.stance_stability_time += self.dt
            reward += self.techniques['stance']['reward'] * self.dt
            
        # Energy efficiency penalty
        energy_cost = np.sum(np.abs(action)) * 0.01
        reward -= energy_cost
        
        # Distance to active dummy (encourage approaching target)
        dummy_pos = self.data.xpos[self.dummy1_idx if self.active_dummy_idx == 0 else self.dummy2_idx]
        torso_pos = self.data.xpos[self.torso_idx]
        distance = np.linalg.norm(dummy_pos[:2] - torso_pos[:2])
        
        if distance < 2.0:
            reward += 50 * (2.0 - distance)
            
        return reward
        
    def _check_termination(self):
        """Check if episode should terminate."""
        # Check if robot has fallen
        torso_height = self.data.xpos[self.torso_idx][2]
        if torso_height < 0.5:
            self.episode_stats['falls'] += 1
            return True
            
        # Check if robot is out of bounds
        torso_pos = self.data.xpos[self.torso_idx]
        if abs(torso_pos[0]) > 5.5 or abs(torso_pos[1]) > 5.5:
            return True
            
        return False
        
    def _get_info(self):
        """Get additional information about the environment state."""
        return {
            'episode_stats': self.episode_stats.copy(),
            'combo_chain': self.combo_chain.copy(),
            'stance_stability': self.stance_stability_time,
            'current_step': self.current_step
        }
        
    def _update_statistics(self):
        """Update episode statistics."""
        # Track total distance moved
        if self.current_step > 1:
            torso_pos = self.data.xpos[self.torso_idx]
            if hasattr(self, 'prev_torso_pos'):
                distance = np.linalg.norm(torso_pos[:2] - self.prev_torso_pos[:2])
                self.episode_stats['total_distance_moved'] += distance
            self.prev_torso_pos = torso_pos.copy()
            
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()
            
    def close(self):
        """Clean up resources."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
