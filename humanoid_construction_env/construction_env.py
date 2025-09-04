"""
Humanoid Construction Environment - A MuJoCo-based 3D construction simulation

This environment simulates a humanoid robot working in a construction site,
performing tasks like crane operation, block stacking, and material transport.
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


class HumanoidConstructionEnv(gym.Env):
    """
    A MuJoCo-based humanoid construction environment.
    
    The robot must learn to control a 27-DOF humanoid to perform construction tasks,
    including crane operation, building structures, and transporting materials.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 50
    }
    
    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        """Initialize the humanoid construction environment."""
        super().__init__()
        
        # Environment parameters
        self.dt = 0.02  # 50 Hz simulation
        self.max_episode_steps = 3000  # 60 seconds at 50 Hz
        self.current_step = 0
        
        # Construction site dimensions (meters)
        self.site_width = 40.0
        self.site_length = 40.0
        self.crane_height = 15.0
        self.building_area = {'x': [-10, 10], 'y': [-10, 10]}
        
        # Construction blocks properties
        self.block_sizes = {
            'small': [0.5, 0.5, 0.25],
            'medium': [1.0, 0.5, 0.25],
            'large': [2.0, 1.0, 0.5]
        }
        self.max_blocks = 20
        self.blocks_placed = 0
        
        # Materials properties
        self.material_types = ['concrete', 'steel', 'wood']
        self.material_weights = {'concrete': 50.0, 'steel': 80.0, 'wood': 30.0}
        
        # Robot properties
        self.robot_height = 1.8  # Taller for construction work
        self.num_joints = 27  # Extra joints for fine manipulation
        self.grip_strength = 500.0  # Newton force for gripping
        
        # Crane properties
        self.crane_angle = 0.0
        self.crane_extension = 5.0
        self.crane_hook_height = 10.0
        self.crane_max_load = 200.0  # kg
        
        # Task parameters
        self.current_task = None
        self.task_types = ['stack_blocks', 'operate_crane', 'transport_material', 'build_structure']
        self.task_progress = 0.0
        
        # Reward parameters
        self.task_completion_reward = 5000.0
        self.block_placed_reward = 500.0
        self.material_transported_reward = 300.0
        self.crane_operation_reward = 200.0
        self.safety_bonus = 100.0
        self.stability_reward = 50.0
        self.energy_penalty = -0.2
        self.collision_penalty = -500.0
        self.fall_penalty = -2000.0
        self.drop_penalty = -1000.0
        
        # Safety parameters
        self.safety_violations = 0
        self.hard_hat_on = True
        self.safety_zone_radius = 3.0
        
        # Weather conditions
        self.wind_strength = 0.0
        self.rain_intensity = 0.0
        self.temperature = 20.0  # Celsius
        
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
        
        # Initialize viewer for rendering (using proven mujoco.viewer approach)
        self.viewer = None
        if self.render_mode == "human":
            self._init_viewer()
        
        # Episode tracking
        self.episode_stats = {
            'blocks_placed': 0,
            'materials_transported': 0,
            'crane_operations': 0,
            'safety_violations': 0,
            'tasks_completed': 0,
            'total_reward': 0.0
        }
        
        # Set random seed
        self.seed()
        
    def _load_xml_models(self):
        """Load and combine XML model files."""
        # Get the directory containing the XML files
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        
        # Create assets directory if it doesn't exist
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)
            
        # Generate XML files if they don't exist
        self._generate_xml_files(assets_dir)
        
        # Load main model files
        site_path = os.path.join(assets_dir, 'construction_site.xml')
        humanoid_path = os.path.join(assets_dir, 'construction_humanoid.xml')
        crane_path = os.path.join(assets_dir, 'crane.xml')
        
        # Read and combine XML files
        with open(site_path, 'r') as f:
            site_xml = f.read()
            
        # For now, use the site XML as the base
        # In a full implementation, you would combine multiple XMLs
        self.xml_string = site_xml
        
    def _generate_xml_files(self, assets_dir):
        """Generate XML model files if they don't exist."""
        # Generate construction site XML
        site_xml = self._generate_site_xml()
        with open(os.path.join(assets_dir, 'construction_site.xml'), 'w') as f:
            f.write(site_xml)
            
        # Generate humanoid XML
        humanoid_xml = self._generate_humanoid_xml()
        with open(os.path.join(assets_dir, 'construction_humanoid.xml'), 'w') as f:
            f.write(humanoid_xml)
            
        # Generate crane XML
        crane_xml = self._generate_crane_xml()
        with open(os.path.join(assets_dir, 'crane.xml'), 'w') as f:
            f.write(crane_xml)
            
    def _generate_site_xml(self):
        """Generate the construction site XML model."""
        return '''<mujoco model="construction_site">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
    
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="3" density="5.0" friction="1.5 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
        <motor ctrllimited="true" ctrlrange="-200.0 200.0"/>
    </default>
    
    <option gravity="0 0 -9.81" integrator="RK4" timestep="0.002"/>
    
    <asset>
        <texture builtin="gradient" height="100" rgb1="0.4 0.6 0.8" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="128" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0.2 0.2 0.2" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.3" shininess="1" specular="1" texrepeat="30 30" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
        <material name="concrete" rgba="0.7 0.7 0.7 1" shininess="0.3"/>
        <material name="steel" rgba="0.8 0.8 0.9 1" shininess="0.8" specular="0.9"/>
        <material name="wood" rgba="0.6 0.4 0.2 1" shininess="0.2"/>
        <material name="safety" rgba="1 0.5 0 1" shininess="0.5"/>
    </asset>
    
    <worldbody>
        <!-- Ground plane -->
        <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" size="40 40 0.1" type="plane"/>
        
        <!-- Construction site boundaries -->
        <body name="site_boundaries">
            <!-- Safety fence around construction site -->
            <geom name="fence_north" pos="0 20 1" size="20 0.1 1" type="box" rgba="1 0.5 0 0.8"/>
            <geom name="fence_south" pos="0 -20 1" size="20 0.1 1" type="box" rgba="1 0.5 0 0.8"/>
            <geom name="fence_east" pos="20 0 1" size="0.1 20 1" type="box" rgba="1 0.5 0 0.8"/>
            <geom name="fence_west" pos="-20 0 1" size="0.1 20 1" type="box" rgba="1 0.5 0 0.8"/>
        </body>
        
        <!-- Foundation area -->
        <body name="foundation">
            <geom name="foundation_base" pos="0 0 0.05" size="10 10 0.05" type="box" material="concrete"/>
            <!-- Foundation pillars -->
            <geom name="pillar1" pos="-8 -8 0.5" size="0.5 0.5 0.5" type="box" material="concrete"/>
            <geom name="pillar2" pos="8 -8 0.5" size="0.5 0.5 0.5" type="box" material="concrete"/>
            <geom name="pillar3" pos="-8 8 0.5" size="0.5 0.5 0.5" type="box" material="concrete"/>
            <geom name="pillar4" pos="8 8 0.5" size="0.5 0.5 0.5" type="box" material="concrete"/>
        </body>
        
        <!-- Humanoid robot with enhanced construction capabilities -->
        <body name="humanoid" pos="5 0 1.5">
            <freejoint name="humanoid_root"/>
            <camera name="track" mode="trackcom" pos="0 -5 2" xyaxes="1 0 0 0 0 1"/>
            
            <!-- Torso with tool belt -->
            <geom name="torso" pos="0 0 0" size="0.35 0.25 0.3" type="capsule" rgba="0.3 0.3 0.8 1"/>
            <geom name="tool_belt" pos="0 0 -0.2" size="0.4 0.15 0.05" type="box" rgba="0.4 0.2 0.1 1"/>
            
            <!-- Head with hard hat -->
            <body name="head" pos="0 0 0.35">
                <geom name="head" pos="0 0 0.1" size="0.12 0.12 0.1" type="capsule" rgba="0.8 0.6 0.4 1"/>
                <geom name="hard_hat" pos="0 0 0.15" size="0.15 0.15 0.08" type="box" material="safety"/>
                <camera name="head_camera" pos="0.1 0 0.05" euler="0 0 0" fovy="60"/>
            </body>
            
            <!-- Arms with enhanced grippers -->
            <!-- Right arm -->
            <body name="right_upper_arm" pos="0.4 0 0.15">
                <joint name="right_shoulder1" type="hinge" axis="1 0 0" range="-2.5 2.5"/>
                <joint name="right_shoulder2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <joint name="right_shoulder3" type="hinge" axis="0 0 1" range="-2 2"/>
                <geom name="right_upper_arm" pos="0.2 0 0" size="0.05 0.2" type="capsule" rgba="0.3 0.3 0.8 1"/>
                
                <body name="right_lower_arm" pos="0.4 0 0">
                    <joint name="right_elbow" type="hinge" axis="0 1 0" range="-2.5 0"/>
                    <geom name="right_lower_arm" pos="0.2 0 0" size="0.04 0.2" type="capsule" rgba="0.3 0.3 0.8 1"/>
                    
                    <!-- Enhanced gripper hand -->
                    <body name="right_hand" pos="0.4 0 0">
                        <joint name="right_wrist1" type="hinge" axis="1 0 0" range="-1 1"/>
                        <joint name="right_wrist2" type="hinge" axis="0 1 0" range="-1 1"/>
                        <geom name="right_palm" pos="0 0 0" size="0.06 0.03 0.08" type="box" rgba="0.8 0.6 0.4 1"/>
                        
                        <!-- Gripper fingers -->
                        <body name="right_thumb" pos="0.06 0 0">
                            <joint name="right_thumb_joint" type="hinge" axis="0 1 0" range="0 1.5"/>
                            <geom name="right_thumb" pos="0.02 0 0" size="0.015 0.03" type="capsule" rgba="0.8 0.6 0.4 1"/>
                        </body>
                        <body name="right_finger1" pos="-0.03 0.02 0">
                            <joint name="right_finger1_joint" type="hinge" axis="0 1 0" range="0 1.5"/>
                            <geom name="right_finger1" pos="-0.02 0 0" size="0.01 0.025" type="capsule" rgba="0.8 0.6 0.4 1"/>
                        </body>
                        <body name="right_finger2" pos="-0.03 -0.02 0">
                            <joint name="right_finger2_joint" type="hinge" axis="0 1 0" range="0 1.5"/>
                            <geom name="right_finger2" pos="-0.02 0 0" size="0.01 0.025" type="capsule" rgba="0.8 0.6 0.4 1"/>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- Left arm (mirror of right) -->
            <body name="left_upper_arm" pos="-0.4 0 0.15">
                <joint name="left_shoulder1" type="hinge" axis="1 0 0" range="-2.5 2.5"/>
                <joint name="left_shoulder2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <joint name="left_shoulder3" type="hinge" axis="0 0 1" range="-2 2"/>
                <geom name="left_upper_arm" pos="-0.2 0 0" size="0.05 0.2" type="capsule" rgba="0.3 0.3 0.8 1"/>
                
                <body name="left_lower_arm" pos="-0.4 0 0">
                    <joint name="left_elbow" type="hinge" axis="0 1 0" range="-2.5 0"/>
                    <geom name="left_lower_arm" pos="-0.2 0 0" size="0.04 0.2" type="capsule" rgba="0.3 0.3 0.8 1"/>
                    
                    <!-- Enhanced gripper hand -->
                    <body name="left_hand" pos="-0.4 0 0">
                        <joint name="left_wrist1" type="hinge" axis="1 0 0" range="-1 1"/>
                        <joint name="left_wrist2" type="hinge" axis="0 1 0" range="-1 1"/>
                        <geom name="left_palm" pos="0 0 0" size="0.06 0.03 0.08" type="box" rgba="0.8 0.6 0.4 1"/>
                        
                        <!-- Gripper fingers -->
                        <body name="left_thumb" pos="-0.06 0 0">
                            <joint name="left_thumb_joint" type="hinge" axis="0 1 0" range="0 1.5"/>
                            <geom name="left_thumb" pos="-0.02 0 0" size="0.015 0.03" type="capsule" rgba="0.8 0.6 0.4 1"/>
                        </body>
                        <body name="left_finger1" pos="0.03 0.02 0">
                            <joint name="left_finger1_joint" type="hinge" axis="0 1 0" range="0 1.5"/>
                            <geom name="left_finger1" pos="0.02 0 0" size="0.01 0.025" type="capsule" rgba="0.8 0.6 0.4 1"/>
                        </body>
                        <body name="left_finger2" pos="0.03 -0.02 0">
                            <joint name="left_finger2_joint" type="hinge" axis="0 1 0" range="0 1.5"/>
                            <geom name="left_finger2" pos="0.02 0 0" size="0.01 0.025" type="capsule" rgba="0.8 0.6 0.4 1"/>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- Strong legs for carrying heavy materials -->
            <!-- Right leg -->
            <body name="right_thigh" pos="0.2 0 -0.4">
                <joint name="right_hip1" type="hinge" axis="1 0 0" range="-2 1"/>
                <joint name="right_hip2" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                <joint name="right_hip3" type="hinge" axis="0 0 1" range="-1 1"/>
                <geom name="right_thigh" pos="0 0 -0.25" size="0.08 0.25" type="capsule" rgba="0.3 0.3 0.8 1"/>
                
                <body name="right_shin" pos="0 0 -0.5">
                    <joint name="right_knee" type="hinge" axis="0 1 0" range="0 2.5"/>
                    <geom name="right_shin" pos="0 0 -0.2" size="0.06 0.2" type="capsule" rgba="0.3 0.3 0.8 1"/>
                    
                    <!-- Safety boot -->
                    <body name="right_foot" pos="0 0 -0.4">
                        <joint name="right_ankle1" type="hinge" axis="1 0 0" range="-1 1"/>
                        <joint name="right_ankle2" type="hinge" axis="0 1 0" range="-1 1"/>
                        <geom name="right_foot" pos="0 0.05 -0.02" size="0.08 0.15 0.03" type="box" rgba="0.2 0.2 0.2 1"/>
                        <geom name="right_boot_steel" pos="0 0.1 -0.02" size="0.08 0.05 0.04" type="box" material="steel"/>
                    </body>
                </body>
            </body>
            
            <!-- Left leg (mirror of right) -->
            <body name="left_thigh" pos="-0.2 0 -0.4">
                <joint name="left_hip1" type="hinge" axis="1 0 0" range="-2 1"/>
                <joint name="left_hip2" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                <joint name="left_hip3" type="hinge" axis="0 0 1" range="-1 1"/>
                <geom name="left_thigh" pos="0 0 -0.25" size="0.08 0.25" type="capsule" rgba="0.3 0.3 0.8 1"/>
                
                <body name="left_shin" pos="0 0 -0.5">
                    <joint name="left_knee" type="hinge" axis="0 1 0" range="0 2.5"/>
                    <geom name="left_shin" pos="0 0 -0.2" size="0.06 0.2" type="capsule" rgba="0.3 0.3 0.8 1"/>
                    
                    <!-- Safety boot -->
                    <body name="left_foot" pos="0 0 -0.4">
                        <joint name="left_ankle1" type="hinge" axis="1 0 0" range="-1 1"/>
                        <joint name="left_ankle2" type="hinge" axis="0 1 0" range="-1 1"/>
                        <geom name="left_foot" pos="0 0.05 -0.02" size="0.08 0.15 0.03" type="box" rgba="0.2 0.2 0.2 1"/>
                        <geom name="left_boot_steel" pos="0 0.1 -0.02" size="0.08 0.05 0.04" type="box" material="steel"/>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- Construction crane -->
        <body name="crane_base" pos="-15 0 0">
            <geom name="crane_foundation" pos="0 0 0.5" size="2 2 0.5" type="box" material="concrete"/>
            <joint name="crane_rotation" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="10"/>
            
            <!-- Crane tower -->
            <geom name="crane_tower" pos="0 0 8" size="0.5 0.5 7.5" type="box" material="steel"/>
            
            <!-- Crane arm -->
            <body name="crane_arm" pos="0 0 15">
                <joint name="crane_arm_angle" type="hinge" axis="0 1 0" range="-0.5 0.5" damping="5"/>
                <geom name="crane_arm_beam" pos="10 0 0" size="10 0.3 0.3" type="box" material="steel"/>
                
                <!-- Crane hook -->
                <body name="crane_hook" pos="20 0 0">
                    <joint name="crane_cable" type="slide" axis="0 0 1" range="-14 0" damping="2"/>
                    <geom name="crane_hook_block" pos="0 0 0" size="0.2 0.2 0.3" type="box" material="steel"/>
                    <geom name="crane_hook_tip" pos="0 0 -0.4" size="0.15 0.15 0.1" type="box" material="steel"/>
                </body>
            </body>
        </body>
        
        <!-- Construction blocks (each at top level with freejoint) -->
        <!-- Small blocks -->
        <body name="small_block1" pos="10 10 0.25">
            <freejoint/>
            <geom name="small_block1_geom" size="0.25 0.25 0.125" type="box" material="concrete" mass="10"/>
        </body>
        <body name="small_block2" pos="10.6 10 0.25">
            <freejoint/>
            <geom name="small_block2_geom" size="0.25 0.25 0.125" type="box" material="concrete" mass="10"/>
        </body>
        <body name="small_block3" pos="11.2 10 0.25">
            <freejoint/>
            <geom name="small_block3_geom" size="0.25 0.25 0.125" type="box" material="concrete" mass="10"/>
        </body>
        
        <!-- Medium blocks -->
        <body name="medium_block1" pos="10 11 0.25">
            <freejoint/>
            <geom name="medium_block1_geom" size="0.5 0.25 0.125" type="box" material="concrete" mass="20"/>
        </body>
        <body name="medium_block2" pos="11.2 11 0.25">
            <freejoint/>
            <geom name="medium_block2_geom" size="0.5 0.25 0.125" type="box" material="concrete" mass="20"/>
        </body>
        
        <!-- Large blocks -->
        <body name="large_block1" pos="10 12 0.5">
            <freejoint/>
            <geom name="large_block1_geom" size="1 0.5 0.25" type="box" material="concrete" mass="40"/>
        </body>
        
        <!-- Material storage area -->
        <body name="material_storage" pos="15 -10 0">
            <geom name="storage_platform" pos="0 0 0.1" size="5 5 0.1" type="box" rgba="0.5 0.5 0.5 1"/>
        </body>
        
        <!-- Steel beams (at top level) -->
        <body name="steel_beam1" pos="13 -10 0.3">
            <freejoint/>
            <geom name="steel_beam1_geom" size="0.15 0.15 2" type="box" material="steel" mass="50"/>
        </body>
        <body name="steel_beam2" pos="14 -10 0.3">
            <freejoint/>
            <geom name="steel_beam2_geom" size="0.15 0.15 2" type="box" material="steel" mass="50"/>
        </body>
        
        <!-- Wood planks (at top level) -->
        <body name="wood_plank1" pos="16 -10 0.2">
            <freejoint/>
            <geom name="wood_plank1_geom" size="1 0.2 0.05" type="box" material="wood" mass="5"/>
        </body>
        <body name="wood_plank2" pos="16 -9.5 0.2">
            <freejoint/>
            <geom name="wood_plank2_geom" size="1 0.2 0.05" type="box" material="wood" mass="5"/>
        </body>
        
        <!-- Safety equipment station -->
        <body name="safety_station" pos="-10 -15 0">
            <geom name="safety_station_base" pos="0 0 0.5" size="1 1 0.5" type="box" material="safety"/>
            <geom name="safety_sign" pos="0 0 2" size="0.5 0.05 0.5" type="box" rgba="1 1 0 1"/>
        </body>
        
        <!-- Construction target area markers -->
        <body name="target_markers">
            <geom name="target1" pos="0 5 0.01" size="1 1 0.01" type="box" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>
            <geom name="target2" pos="0 7 0.01" size="1 1 0.01" type="box" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>
            <geom name="target3" pos="0 9 0.01" size="1 1 0.01" type="box" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>
        </body>
        
        <!-- Lighting for construction site -->
        <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true" exponent="1" pos="0 0 30" specular="0.5 0.5 0.5"/>
        <light cutoff="50" diffuse="0.8 0.8 0.8" dir="1 0 -0.5" directional="true" pos="-20 0 20" specular="0.3 0.3 0.3"/>
        <light cutoff="50" diffuse="0.8 0.8 0.8" dir="-1 0 -0.5" directional="true" pos="20 0 20" specular="0.3 0.3 0.3"/>
    </worldbody>
    
    <actuator>
        <!-- Humanoid actuators with construction-appropriate strength -->
        <!-- Arms -->
        <motor name="right_shoulder1_motor" joint="right_shoulder1" gear="100"/>
        <motor name="right_shoulder2_motor" joint="right_shoulder2" gear="100"/>
        <motor name="right_shoulder3_motor" joint="right_shoulder3" gear="100"/>
        <motor name="right_elbow_motor" joint="right_elbow" gear="80"/>
        <motor name="right_wrist1_motor" joint="right_wrist1" gear="50"/>
        <motor name="right_wrist2_motor" joint="right_wrist2" gear="50"/>
        <motor name="right_thumb_motor" joint="right_thumb_joint" gear="30"/>
        <motor name="right_finger1_motor" joint="right_finger1_joint" gear="30"/>
        <motor name="right_finger2_motor" joint="right_finger2_joint" gear="30"/>
        
        <motor name="left_shoulder1_motor" joint="left_shoulder1" gear="100"/>
        <motor name="left_shoulder2_motor" joint="left_shoulder2" gear="100"/>
        <motor name="left_shoulder3_motor" joint="left_shoulder3" gear="100"/>
        <motor name="left_elbow_motor" joint="left_elbow" gear="80"/>
        <motor name="left_wrist1_motor" joint="left_wrist1" gear="50"/>
        <motor name="left_wrist2_motor" joint="left_wrist2" gear="50"/>
        <motor name="left_thumb_motor" joint="left_thumb_joint" gear="30"/>
        <motor name="left_finger1_motor" joint="left_finger1_joint" gear="30"/>
        <motor name="left_finger2_motor" joint="left_finger2_joint" gear="30"/>
        
        <!-- Legs -->
        <motor name="right_hip1_motor" joint="right_hip1" gear="150"/>
        <motor name="right_hip2_motor" joint="right_hip2" gear="150"/>
        <motor name="right_hip3_motor" joint="right_hip3" gear="150"/>
        <motor name="right_knee_motor" joint="right_knee" gear="120"/>
        <motor name="right_ankle1_motor" joint="right_ankle1" gear="80"/>
        <motor name="right_ankle2_motor" joint="right_ankle2" gear="80"/>
        
        <motor name="left_hip1_motor" joint="left_hip1" gear="150"/>
        <motor name="left_hip2_motor" joint="left_hip2" gear="150"/>
        <motor name="left_hip3_motor" joint="left_hip3" gear="150"/>
        <motor name="left_knee_motor" joint="left_knee" gear="120"/>
        <motor name="left_ankle1_motor" joint="left_ankle1" gear="80"/>
        <motor name="left_ankle2_motor" joint="left_ankle2" gear="80"/>
        
        <!-- Crane controls -->
        <motor name="crane_rotation_motor" joint="crane_rotation" gear="500"/>
        <motor name="crane_arm_motor" joint="crane_arm_angle" gear="300"/>
        <motor name="crane_cable_motor" joint="crane_cable" gear="200"/>
    </actuator>
</mujoco>'''
        
    def _generate_humanoid_xml(self):
        """Generate the construction humanoid XML model."""
        # This would contain just the humanoid portion for modular loading
        return '''<mujoco model="construction_humanoid">
    <!-- Construction-specific humanoid model -->
</mujoco>'''
        
    def _generate_crane_xml(self):
        """Generate the crane XML model."""
        # This would contain just the crane portion for modular loading
        return '''<mujoco model="crane">
    <!-- Construction crane model -->
</mujoco>'''
        
    def _get_model_indices(self):
        """Get indices for important model elements."""
        # Get body indices
        self.humanoid_id = self.model.body('humanoid').id if self.model.nbody > 0 else 0
        self.crane_id = self.model.body('crane_base').id if self.model.nbody > 0 else 0
        
        # Get joint indices (simplified for now)
        self.joint_indices = list(range(self.model.njnt))
        
    def _define_spaces(self):
        """Define action and observation spaces."""
        # Action space: Joint torques for humanoid + crane controls
        action_low = -200.0 * np.ones(self.num_joints)
        action_high = 200.0 * np.ones(self.num_joints)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # Observation space
        obs_dim = 125  # Comprehensive observation including:
        # - Robot proprioception (joint angles, velocities): ~50
        # - Object positions and states: ~30
        # - Task information: ~10
        # - Environment state: ~10
        # - Safety indicators: ~5
        # - Construction progress: ~20
        
        obs_low = -np.inf * np.ones(obs_dim)
        obs_high = np.inf * np.ones(obs_dim)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
    def _init_viewer(self):
        """Initialize the viewer for rendering (using proven mujoco.viewer approach)."""
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 20.0
        self.viewer.cam.elevation = -20.0
        self.viewer.cam.azimuth = 135.0
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset episode tracking
        self.current_step = 0
        self.blocks_placed = 0
        self.safety_violations = 0
        
        # Select random task
        self.current_task = self.np_random.choice(self.task_types)
        self.task_progress = 0.0
        
        # Reset episode stats
        self.episode_stats = {
            'blocks_placed': 0,
            'materials_transported': 0,
            'crane_operations': 0,
            'safety_violations': 0,
            'tasks_completed': 0,
            'total_reward': 0.0
        }
        
        # Set random weather conditions
        self.wind_strength = self.np_random.uniform(0, 5)
        self.rain_intensity = self.np_random.uniform(0, 0.5)
        self.temperature = self.np_random.uniform(15, 35)
        
        # Get initial observation
        obs = self._get_observation()
        
        # Get info
        info = self._get_info()
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply actions to actuators
        self.data.ctrl[:] = action
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        
        # Update step counter
        self.current_step += 1
        
        # Check task progress
        self._update_task_progress()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        # Get observation
        obs = self._get_observation()
        
        # Get info
        info = self._get_info()
        
        # Update episode stats
        self.episode_stats['total_reward'] += reward
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        obs = []
        
        # Robot proprioception
        obs.extend(self.data.qpos[:30])  # Joint positions
        obs.extend(self.data.qvel[:30])  # Joint velocities
        
        # Object states (simplified)
        obs.extend([0.0] * 30)  # Placeholder for object positions
        
        # Task information
        task_one_hot = [0.0] * len(self.task_types)
        task_idx = self.task_types.index(self.current_task)
        task_one_hot[task_idx] = 1.0
        obs.extend(task_one_hot)
        obs.append(self.task_progress)
        obs.extend([0.0] * 5)  # Task-specific parameters
        
        # Environment state
        obs.append(self.wind_strength / 10.0)
        obs.append(self.rain_intensity)
        obs.append(self.temperature / 50.0)
        obs.extend([0.0] * 7)  # Additional environment features
        
        # Safety indicators
        obs.append(float(self.hard_hat_on))
        obs.append(float(self.safety_violations) / 10.0)
        obs.extend([0.0] * 3)  # Additional safety features
        
        # Construction progress
        obs.append(float(self.blocks_placed) / self.max_blocks)
        obs.extend([0.0] * 19)  # Additional progress features
        
        return np.array(obs, dtype=np.float32)
        
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate the reward for the current state and action."""
        reward = 0.0
        
        # Task-specific rewards
        if self.current_task == 'stack_blocks':
            # Reward for getting close to blocks
            # Reward for successfully stacking
            reward += self.task_progress * self.block_placed_reward
            
        elif self.current_task == 'operate_crane':
            # Reward for crane operation
            reward += self.crane_operation_reward * 0.1
            
        elif self.current_task == 'transport_material':
            # Reward for material transport
            reward += self.material_transported_reward * 0.1
            
        elif self.current_task == 'build_structure':
            # Reward for building progress
            reward += self.task_progress * 100
            
        # Safety bonus for wearing hard hat
        if self.hard_hat_on:
            reward += self.safety_bonus * 0.01
            
        # Penalty for safety violations
        reward -= self.safety_violations * 100
        
        # Energy penalty (encourage efficient movements)
        reward += self.energy_penalty * np.sum(np.abs(action))
        
        # Stability reward (robot staying upright)
        robot_height = self.data.xpos[self.humanoid_id][2] if self.humanoid_id < len(self.data.xpos) else 1.5
        if robot_height > 1.0:
            reward += self.stability_reward * 0.1
        else:
            reward += self.fall_penalty
            
        return reward
        
    def _update_task_progress(self):
        """Update progress on current task."""
        # Simplified task progress update
        if self.current_task == 'stack_blocks':
            # Check if blocks are stacked properly
            self.task_progress = min(1.0, self.blocks_placed / 5)
            
        elif self.current_task == 'operate_crane':
            # Check crane operation metrics
            self.task_progress = min(1.0, self.current_step / 500)
            
        elif self.current_task == 'transport_material':
            # Check material transport
            self.task_progress = min(1.0, self.current_step / 300)
            
        elif self.current_task == 'build_structure':
            # Check building progress
            self.task_progress = min(1.0, self.blocks_placed / 10)
            
    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Check if robot has fallen
        robot_height = self.data.xpos[self.humanoid_id][2] if self.humanoid_id < len(self.data.xpos) else 1.5
        if robot_height < 0.5:
            return True
            
        # Check if task is completed
        if self.task_progress >= 1.0:
            self.episode_stats['tasks_completed'] += 1
            return True
            
        # Check for critical safety violations
        if self.safety_violations > 3:
            return True
            
        return False
        
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            'task': self.current_task,
            'task_progress': self.task_progress,
            'blocks_placed': self.blocks_placed,
            'safety_violations': self.safety_violations,
            'episode_stats': self.episode_stats.copy(),
            'weather': {
                'wind': self.wind_strength,
                'rain': self.rain_intensity,
                'temperature': self.temperature
            }
        }
        
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()
            
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
