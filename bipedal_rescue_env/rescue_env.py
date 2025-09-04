"""
Bipedal Rescue Environment - A MuJoCo-based disaster rescue simulation

This environment simulates a bipedal robot navigating through a disaster scene,
rescuing victims and bringing them to safety zones.

Author: Hasnain Fareed
License: MIT (2025)
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


class BipedalRescueEnv(gym.Env):
    """
    A MuJoCo-based bipedal robot rescue environment.
    
    The robot must navigate through debris, avoid hazards, locate victims,
    and carry them to safety zones while maintaining balance.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 50
    }
    
    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        """Initialize the bipedal rescue environment."""
        super().__init__()
        
        # Environment parameters
        self.dt = 0.02  # 50 Hz simulation
        self.max_episode_steps = 10000  # 200 seconds at 50 Hz
        self.current_step = 0
        
        # Environment dimensions
        self.world_size = 50.0  # 50m x 50m world
        self.safe_zone_radius = 3.0
        self.safe_zone_pos = np.array([20.0, 0.0, 0.0])
        
        # Robot parameters
        self.robot_height = 1.2
        self.carry_capacity = 2  # Can carry up to 2 victims
        self.energy_limit = 1000.0
        self.current_energy = self.energy_limit
        
        # Victim parameters
        self.num_victims = 5
        self.victim_weights = [60, 30, 50, 55, 65]  # kg
        self.victim_priorities = [0.8, 1.0, 0.7, 0.9, 1.0]  # Priority based on injury severity
        self.victims_rescued = []
        self.victims_carried = []
        
        # Reward parameters
        self.victim_rescue_reward = 5000.0
        self.victim_pickup_reward = 1000.0
        self.approach_victim_reward = 100.0
        self.safe_zone_approach_reward = 200.0
        self.balance_reward = 50.0
        self.energy_efficiency_reward = 10.0
        self.collision_penalty = -100.0
        self.fall_penalty = -500.0
        self.hazard_penalty = -200.0
        self.time_penalty = -1.0
        
        # Hazard zones
        self.fire_zones = [
            {'pos': np.array([-5.0, -3.0, 0.0]), 'radius': 1.5},
            {'pos': np.array([8.0, 6.0, 0.0]), 'radius': 1.2}
        ]
        
        # Set render mode
        self.render_mode = render_mode
        
        # Load and combine XML models
        self._load_xml_models()
        
        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)
        
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
            'victims_rescued': 0,
            'distance_traveled': 0.0,
            'energy_used': 0.0,
            'time_to_first_rescue': None,
            'falls': 0,
            'collisions': 0
        }
        
        # Previous state for reward calculation
        self.prev_robot_pos = None
        self.prev_robot_quat = None
        self.closest_victim_distance = float('inf')
        self.carrying_victims = False
        
        # Seed for reproducibility
        self.np_random = None
        self.seed()
    
    def _load_xml_models(self):
        """Load and combine all XML model files into a single model."""
        
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, 'assets')
        
        # Load individual XML files
        robot_path = os.path.join(assets_dir, 'bipedal_robot.xml')
        environment_path = os.path.join(assets_dir, 'disaster_environment.xml')
        victims_path = os.path.join(assets_dir, 'victims.xml')
        
        # Create combined model
        root = ET.Element('mujoco', model='bipedal_rescue')
        
        # Add compiler settings
        compiler = ET.SubElement(root, 'compiler')
        compiler.set('angle', 'radian')
        compiler.set('coordinate', 'local')
        compiler.set('inertiafromgeom', 'true')
        
        # Add option settings
        option = ET.SubElement(root, 'option')
        option.set('timestep', str(self.dt))
        option.set('iterations', '50')
        option.set('solver', 'PGS')
        option.set('gravity', '0 0 -9.81')
        option.set('integrator', 'RK4')
        
        # Add visual settings
        visual = ET.SubElement(root, 'visual')
        map_elem = ET.SubElement(visual, 'map')
        map_elem.set('fogstart', '10')
        map_elem.set('fogend', '50')
        quality = ET.SubElement(visual, 'quality')
        quality.set('shadowsize', '2048')
        
        # Combine defaults
        default = ET.SubElement(root, 'default')
        joint_default = ET.SubElement(default, 'joint')
        joint_default.set('armature', '1')
        joint_default.set('damping', '1')
        joint_default.set('limited', 'true')
        
        geom_default = ET.SubElement(default, 'geom')
        geom_default.set('conaffinity', '1')
        geom_default.set('condim', '3')
        geom_default.set('density', '5.0')
        geom_default.set('friction', '1 0.5 0.5')
        geom_default.set('margin', '0.01')
        geom_default.set('rgba', '0.8 0.6 0.4 1')
        
        motor_default = ET.SubElement(default, 'motor')
        motor_default.set('ctrllimited', 'true')
        motor_default.set('ctrlrange', '-100.0 100.0')
        
        # Add assets
        asset = ET.SubElement(root, 'asset')
        texture1 = ET.SubElement(asset, 'texture')
        texture1.set('name', 'grid')
        texture1.set('type', '2d')
        texture1.set('builtin', 'checker')
        texture1.set('width', '512')
        texture1.set('height', '512')
        
        material1 = ET.SubElement(asset, 'material')
        material1.set('name', 'grid')
        material1.set('texture', 'grid')
        material1.set('texrepeat', '10 10')
        material1.set('texuniform', 'true')
        material1.set('reflectance', '0.2')
        
        # Add worldbody (combine from all XMLs)
        worldbody = ET.SubElement(root, 'worldbody')
        
        # Parse and add environment elements
        env_tree = ET.parse(environment_path)
        for child in env_tree.getroot().find('.//worldbody'):
            worldbody.append(child)
        
        # Parse and add victim elements
        victims_tree = ET.parse(victims_path)
        for child in victims_tree.getroot():
            if child.tag == 'body':
                worldbody.append(child)
        
        # Parse and add robot elements
        robot_tree = ET.parse(robot_path)
        for child in robot_tree.getroot():
            if child.tag == 'body':
                worldbody.append(child)
        
        # Add lights from environment
        for child in env_tree.getroot():
            if child.tag == 'light':
                worldbody.append(child)
        
        # Add sensors
        sensor = ET.SubElement(root, 'sensor')
        
        # Torso sensors
        torso_accel = ET.SubElement(sensor, 'accelerometer')
        torso_accel.set('name', 'torso_accel')
        torso_accel.set('site', 'torso_site')
        
        torso_gyro = ET.SubElement(sensor, 'gyro')
        torso_gyro.set('name', 'torso_gyro')
        torso_gyro.set('site', 'torso_site')
        
        # Foot contact sensors
        right_foot_touch = ET.SubElement(sensor, 'touch')
        right_foot_touch.set('name', 'right_foot_touch')
        right_foot_touch.set('site', 'right_foot_contact')
        
        left_foot_touch = ET.SubElement(sensor, 'touch')
        left_foot_touch.set('name', 'left_foot_touch')
        left_foot_touch.set('site', 'left_foot_contact')
        
        # Add actuators
        actuator = ET.SubElement(root, 'actuator')
        
        # Define joint names for actuators
        joint_names = [
            'neck_pitch', 'neck_yaw',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow', 'right_wrist',
            'right_finger1_joint', 'right_finger2_joint',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow', 'left_wrist',
            'left_finger1_joint', 'left_finger2_joint',
            'right_hip_roll', 'right_hip_pitch', 'right_hip_yaw', 'right_knee_joint',
            'right_ankle_pitch', 'right_ankle_roll',
            'left_hip_roll', 'left_hip_pitch', 'left_hip_yaw', 'left_knee_joint',
            'left_ankle_pitch', 'left_ankle_roll'
        ]
        
        for joint_name in joint_names:
            motor = ET.SubElement(actuator, 'motor')
            motor.set('name', f'{joint_name}_motor')
            motor.set('joint', joint_name)
            motor.set('gear', '50')
            motor.set('ctrllimited', 'true')
            motor.set('ctrlrange', '-100 100')
        
        # Add contact pairs for collision detection
        contact = ET.SubElement(root, 'contact')
        
        # Robot-victim contact pairs
        for i in range(1, 6):
            pair = ET.SubElement(contact, 'pair')
            pair.set('geom1', 'right_gripper_base')
            pair.set('geom2', f'victim{i}_torso')
            
            pair = ET.SubElement(contact, 'pair')
            pair.set('geom1', 'left_gripper_base')
            pair.set('geom2', f'victim{i}_torso')
        
        # Convert to string
        self.xml_string = ET.tostring(root, encoding='unicode')
    
    def _get_model_indices(self):
        """Get indices for important model elements."""
        
        # Body indices
        try:
            self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
            self.safe_zone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'safe_zone')
            
            # Victim body indices
            self.victim_ids = []
            for i in range(1, self.num_victims + 1):
                try:
                    victim_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'victim{i}')
                    self.victim_ids.append(victim_id)
                except:
                    print(f"Warning: Victim {i} not found")
        except Exception as e:
            print(f"Warning: Some bodies not found: {e}")
        
        # Joint indices
        self.joint_names = [
            'neck_pitch', 'neck_yaw',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow', 'right_wrist',
            'right_finger1_joint', 'right_finger2_joint',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow', 'left_wrist',
            'left_finger1_joint', 'left_finger2_joint',
            'right_hip_roll', 'right_hip_pitch', 'right_hip_yaw', 'right_knee_joint',
            'right_ankle_pitch', 'right_ankle_roll',
            'left_hip_roll', 'left_hip_pitch', 'left_hip_yaw', 'left_knee_joint',
            'left_ankle_pitch', 'left_ankle_roll'
        ]
        
        self.joint_indices = []
        for name in self.joint_names:
            try:
                idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.joint_indices.append(idx)
            except:
                print(f"Warning: Joint {name} not found")
        
        # Site indices for pickups
        self.victim_pickup_sites = []
        for i in range(1, self.num_victims + 1):
            try:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'victim{i}_pickup')
                self.victim_pickup_sites.append(site_id)
            except:
                print(f"Warning: Victim {i} pickup site not found")
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        
        # Action space: torque control for all actuators
        self.num_actuators = len(self.joint_names)
        self.action_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(self.num_actuators,),
            dtype=np.float32
        )
        
        # Observation space
        obs_dim = (
            self.num_actuators * 2 +  # Joint positions and velocities
            7 +  # Robot position and orientation
            6 +  # Robot velocity (linear and angular)
            4 +  # Foot contact forces
            self.num_victims * 4 +  # Victim positions and states
            3 +  # Safe zone relative position
            1 +  # Energy remaining
            1 +  # Time remaining
            2 +  # Number of victims carried/rescued
            len(self.fire_zones) * 3  # Fire zone positions
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def seed(self, seed: Optional[int] = None) -> list:
        """Seed the environment's random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        
        if seed is not None:
            self.seed(seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset episode tracking
        self.current_step = 0
        self.current_energy = self.energy_limit
        self.victims_rescued = []
        self.victims_carried = []
        self.carrying_victims = False
        self.closest_victim_distance = float('inf')
        
        # Reset episode stats
        self.episode_stats = {
            'victims_rescued': 0,
            'distance_traveled': 0.0,
            'energy_used': 0.0,
            'time_to_first_rescue': None,
            'falls': 0,
            'collisions': 0
        }
        
        # Randomize initial positions
        self._randomize_initial_state()
        
        # Step simulation to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Get initial observation
        obs = self._get_observation()
        
        # Store previous state
        self.prev_robot_pos = self._get_robot_position().copy()
        self.prev_robot_quat = self._get_robot_orientation().copy()
        
        info = {
            'episode_stats': self.episode_stats.copy(),
            'robot_position': self._get_robot_position(),
            'victims_remaining': self.num_victims - len(self.victims_rescued),
            'energy_remaining': self.current_energy
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply actions to actuators
        if len(action) <= len(self.data.ctrl):
            self.data.ctrl[:len(action)] = action
        
        # Update energy consumption
        energy_cost = np.sum(np.abs(action)) * 0.001
        self.current_energy -= energy_cost
        self.episode_stats['energy_used'] += energy_cost
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Update step counter
        self.current_step += 1
        
        # Check for victim interactions
        self._check_victim_interactions()
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # Update episode stats
        self._update_episode_stats()
        
        # Prepare info
        info = {
            'episode_stats': self.episode_stats.copy(),
            'robot_position': self._get_robot_position(),
            'victims_remaining': self.num_victims - len(self.victims_rescued),
            'victims_carried': len(self.victims_carried),
            'energy_remaining': self.current_energy,
            'robot_upright': self._is_robot_upright()
        }
        
        # Update previous state
        self.prev_robot_pos = self._get_robot_position().copy()
        self.prev_robot_quat = self._get_robot_orientation().copy()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _randomize_initial_state(self):
        """Randomize initial positions of robot and victims."""
        
        # Robot starting position (near center but safe from hazards)
        robot_x = self.np_random.uniform(-5.0, 5.0)
        robot_y = self.np_random.uniform(-5.0, 5.0)
        robot_z = 1.2
        
        # Set robot position
        root_joint_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root_x')
        if root_joint_idx >= 0:
            self.data.qpos[self.model.jnt_qposadr[root_joint_idx]] = robot_x
            
        root_joint_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root_y')
        if root_joint_idx >= 0:
            self.data.qpos[self.model.jnt_qposadr[root_joint_idx]] = robot_y
            
        root_joint_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root_z')
        if root_joint_idx >= 0:
            self.data.qpos[self.model.jnt_qposadr[root_joint_idx]] = robot_z
        
        # Slightly randomize victim positions around their default locations
        for i, victim_id in enumerate(self.victim_ids, 1):
            victim_x_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'victim{i}_x')
            victim_y_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'victim{i}_y')
            
            if victim_x_joint >= 0 and victim_y_joint >= 0:
                # Add small random offset to victim positions
                x_offset = self.np_random.uniform(-1.0, 1.0)
                y_offset = self.np_random.uniform(-1.0, 1.0)
                
                current_x = self.data.qpos[self.model.jnt_qposadr[victim_x_joint]]
                current_y = self.data.qpos[self.model.jnt_qposadr[victim_y_joint]]
                
                self.data.qpos[self.model.jnt_qposadr[victim_x_joint]] = current_x + x_offset
                self.data.qpos[self.model.jnt_qposadr[victim_y_joint]] = current_y + y_offset
    
    def _check_victim_interactions(self):
        """Check if robot is near victims for pickup or has reached safe zone for drop-off."""
        
        robot_pos = self._get_robot_position()
        
        # Check for victim pickup
        if len(self.victims_carried) < self.carry_capacity:
            for i, victim_id in enumerate(self.victim_ids):
                if i not in self.victims_rescued and i not in self.victims_carried:
                    victim_pos = self._get_victim_position(i)
                    distance = np.linalg.norm(robot_pos[:2] - victim_pos[:2])
                    
                    # Check if close enough to pick up
                    if distance < 1.0 and self._check_gripper_contact(i):
                        self.victims_carried.append(i)
                        self.carrying_victims = True
                        print(f"Picked up victim {i+1}")
        
        # Check for safe zone drop-off
        if self.carrying_victims:
            safe_zone_distance = np.linalg.norm(robot_pos[:2] - self.safe_zone_pos[:2])
            if safe_zone_distance < self.safe_zone_radius:
                # Drop off all carried victims
                for victim_idx in self.victims_carried:
                    self.victims_rescued.append(victim_idx)
                    self.episode_stats['victims_rescued'] += 1
                    
                    if self.episode_stats['time_to_first_rescue'] is None:
                        self.episode_stats['time_to_first_rescue'] = self.current_step * self.dt
                    
                    print(f"Rescued victim {victim_idx+1}")
                
                self.victims_carried = []
                self.carrying_victims = False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        
        obs = []
        
        # Joint positions and velocities
        for joint_idx in self.joint_indices:
            if joint_idx < len(self.data.qpos):
                pos = self.data.qpos[self.model.jnt_qposadr[joint_idx]]
                vel = self.data.qvel[self.model.jnt_dofadr[joint_idx]] if self.model.jnt_dofadr[joint_idx] < len(self.data.qvel) else 0.0
                obs.extend([pos, vel])
            else:
                obs.extend([0.0, 0.0])
        
        # Robot position and orientation
        robot_pos = self._get_robot_position()
        robot_quat = self._get_robot_orientation()
        obs.extend(robot_pos)
        obs.extend(robot_quat)
        
        # Robot velocity
        robot_vel = self._get_robot_velocity()
        obs.extend(robot_vel)
        
        # Foot contact forces
        contact_forces = self._get_foot_contact_forces()
        obs.extend(contact_forces)
        
        # Victim positions and states
        for i in range(self.num_victims):
            victim_pos = self._get_victim_position(i)
            is_rescued = 1.0 if i in self.victims_rescued else 0.0
            is_carried = 1.0 if i in self.victims_carried else 0.0
            priority = self.victim_priorities[i]
            obs.extend([victim_pos[0], victim_pos[1], is_rescued, is_carried])
        
        # Safe zone relative position
        safe_zone_rel = self.safe_zone_pos - robot_pos
        obs.extend(safe_zone_rel)
        
        # Energy remaining (normalized)
        obs.append(self.current_energy / self.energy_limit)
        
        # Time remaining (normalized)
        obs.append(1.0 - (self.current_step / self.max_episode_steps))
        
        # Number of victims carried and rescued
        obs.append(len(self.victims_carried))
        obs.append(len(self.victims_rescued))
        
        # Fire zone positions relative to robot
        for fire_zone in self.fire_zones:
            fire_rel = fire_zone['pos'] - robot_pos
            obs.extend(fire_rel)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current step."""
        
        reward = 0.0
        
        # Victim rescue reward
        if hasattr(self, '_prev_rescued_count'):
            new_rescues = len(self.victims_rescued) - self._prev_rescued_count
            if new_rescues > 0:
                reward += self.victim_rescue_reward * new_rescues
        self._prev_rescued_count = len(self.victims_rescued)
        
        # Victim pickup reward
        if hasattr(self, '_prev_carried_count'):
            new_pickups = len(self.victims_carried) - self._prev_carried_count
            if new_pickups > 0:
                reward += self.victim_pickup_reward * new_pickups
        self._prev_carried_count = len(self.victims_carried)
        
        # Approach victim reward
        robot_pos = self._get_robot_position()
        min_victim_distance = float('inf')
        for i in range(self.num_victims):
            if i not in self.victims_rescued and i not in self.victims_carried:
                victim_pos = self._get_victim_position(i)
                distance = np.linalg.norm(robot_pos[:2] - victim_pos[:2])
                min_victim_distance = min(min_victim_distance, distance)
        
        if min_victim_distance < self.closest_victim_distance and min_victim_distance < 10.0:
            reward += self.approach_victim_reward * (self.closest_victim_distance - min_victim_distance)
        self.closest_victim_distance = min_victim_distance
        
        # Safe zone approach reward (when carrying victims)
        if self.carrying_victims:
            safe_zone_distance = np.linalg.norm(robot_pos[:2] - self.safe_zone_pos[:2])
            if hasattr(self, '_prev_safe_zone_distance'):
                if safe_zone_distance < self._prev_safe_zone_distance:
                    reward += self.safe_zone_approach_reward * (self._prev_safe_zone_distance - safe_zone_distance)
            self._prev_safe_zone_distance = safe_zone_distance
        
        # Balance reward
        if self._is_robot_upright():
            reward += self.balance_reward
        else:
            reward += self.fall_penalty
            self.episode_stats['falls'] += 1
        
        # Energy efficiency reward
        energy_usage = np.sum(np.abs(action)) * 0.001
        if energy_usage < 0.5:  # Low energy usage
            reward += self.energy_efficiency_reward
        
        # Hazard penalty (fire zones)
        for fire_zone in self.fire_zones:
            distance_to_fire = np.linalg.norm(robot_pos[:2] - fire_zone['pos'][:2])
            if distance_to_fire < fire_zone['radius']:
                reward += self.hazard_penalty
        
        # Collision penalty
        if self._check_collision():
            reward += self.collision_penalty
            self.episode_stats['collisions'] += 1
        
        # Time penalty
        reward += self.time_penalty
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        
        # All victims rescued
        if len(self.victims_rescued) == self.num_victims:
            return True
        
        # Robot fell and can't recover
        if not self._is_robot_upright():
            # Check if robot has been down for too long
            if not hasattr(self, '_fall_timer'):
                self._fall_timer = 0
            self._fall_timer += 1
            if self._fall_timer > 100:  # 2 seconds
                return True
        else:
            self._fall_timer = 0
        
        # Energy depleted
        if self.current_energy <= 0:
            return True
        
        # Robot out of bounds
        robot_pos = self._get_robot_position()
        if abs(robot_pos[0]) > 25 or abs(robot_pos[1]) > 25:
            return True
        
        return False
    
    def _update_episode_stats(self):
        """Update episode statistics."""
        
        # Distance traveled
        if self.prev_robot_pos is not None:
            robot_pos = self._get_robot_position()
            distance = np.linalg.norm(robot_pos[:2] - self.prev_robot_pos[:2])
            self.episode_stats['distance_traveled'] += distance
    
    # Helper methods
    def _get_robot_position(self) -> np.ndarray:
        """Get robot torso position."""
        return self.data.xpos[self.torso_id].copy()
    
    def _get_robot_orientation(self) -> np.ndarray:
        """Get robot torso orientation quaternion."""
        return self.data.xquat[self.torso_id].copy()
    
    def _get_robot_velocity(self) -> np.ndarray:
        """Get robot velocity (linear and angular)."""
        root_x_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root_x')
        if root_x_joint >= 0:
            vel_start = self.model.jnt_dofadr[root_x_joint]
            return self.data.qvel[vel_start:vel_start+6].copy()
        return np.zeros(6)
    
    def _get_victim_position(self, victim_idx: int) -> np.ndarray:
        """Get victim position."""
        if victim_idx < len(self.victim_ids):
            return self.data.xpos[self.victim_ids[victim_idx]].copy()
        return np.zeros(3)
    
    def _get_foot_contact_forces(self) -> np.ndarray:
        """Get contact forces on feet."""
        forces = np.zeros(4)
        
        # Simplified contact force calculation
        for i in range(min(self.data.ncon, 10)):
            contact = self.data.contact[i]
            # Check if contact involves foot geoms
            # This is simplified - in practice you'd check specific geom IDs
            forces[0] += abs(contact.dist)
        
        return forces
    
    def _check_gripper_contact(self, victim_idx: int) -> bool:
        """Check if gripper is in contact with victim."""
        # Simplified check - in practice would check actual contacts
        robot_pos = self._get_robot_position()
        victim_pos = self._get_victim_position(victim_idx)
        distance = np.linalg.norm(robot_pos[:2] - victim_pos[:2])
        return distance < 0.8
    
    def _check_collision(self) -> bool:
        """Check if robot collided with obstacles."""
        # Simplified collision check
        for i in range(min(self.data.ncon, 20)):
            contact = self.data.contact[i]
            # Check for high contact forces indicating collision
            if abs(contact.dist) > 0.1:
                return True
        return False
    
    def _is_robot_upright(self) -> bool:
        """Check if robot is upright."""
        
        # Get robot orientation
        quat = self._get_robot_orientation()
        
        # Convert quaternion to rotation matrix
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        
        # Check if z-axis is pointing up
        up_vector = rot_mat[:, 2]
        upright_threshold = 0.7  # cos(45 degrees)
        
        return up_vector[2] > upright_threshold
    
    def _init_viewer(self):
        """Initialize MuJoCo viewer."""
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("🚁 MuJoCo viewer opened successfully!")
        except Exception as e:
            print(f"⚠️ Could not open viewer: {e}")
            self.viewer = None
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
        return None
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Register environment
def register_env():
    """Register the environment with Gymnasium."""
    try:
        gym.register(
            id='BipedalRescue-v0',
            entry_point='bipedal_rescue_env.rescue_env:BipedalRescueEnv',
            max_episode_steps=10000,
            reward_threshold=20000.0,
        )
    except gym.error.Error:
        # Environment already registered
        pass


if __name__ == "__main__":
    # Test the environment
    register_env()
    env = BipedalRescueEnv(render_mode='human')
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    for i in range(1000):
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 50 == 0:
            print(f"Step {i}: Reward = {reward:.2f}, Victims rescued = {info['episode_stats']['victims_rescued']}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            obs, info = env.reset()
    
    env.close()
