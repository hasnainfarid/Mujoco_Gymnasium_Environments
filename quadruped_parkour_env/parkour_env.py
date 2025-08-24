"""
Quadruped Parkour Environment

A MuJoCo-based reinforcement learning environment featuring a quadruped robot
navigating through a challenging parkour course with dynamic obstacles.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
import random
import math


class QuadrupedParkourEnv(gym.Env):
    """
    Quadruped Parkour Environment
    
    A 3D parkour course environment where a quadruped robot must navigate
    through various obstacles including stairs, gaps, balance beams, and
    dynamic terrain variations.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 100
    }
    
    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        """Initialize the Quadruped Parkour Environment."""
        
        # Environment parameters
        self.render_mode = render_mode
        self.dt = 0.01  # Control timestep (100Hz)
        self.frame_skip = 10  # MuJoCo runs at 1000Hz
        self.max_episode_steps = 6000  # 60 seconds at 100Hz
        
        # Course parameters
        self.course_length = 100.0
        self.course_width = 20.0
        self.start_pos = np.array([2.0, 0.0, 0.6])
        self.finish_pos = np.array([98.0, 0.0, 0.0])
        
        # Load MuJoCo model
        self._load_model()
        
        # Initialize MuJoCo simulation
        self.data = mujoco.MjData(self.model)
        
        # Robot state tracking
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        
        # Episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_position = self.start_pos.copy()
        self.max_forward_progress = 0.0
        self.checkpoints_reached = set()
        self.fall_count = 0
        self.stuck_counter = 0
        
        # Obstacle tracking
        self.obstacle_positions = self._get_obstacle_positions()
        self.checkpoint_positions = [15, 30, 45, 60, 75, 90]
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Rendering
        self.viewer = None
        if self.render_mode == 'human':
            self._init_viewer()
        
        # Seeding
        self.np_random = None
        self.seed()
    
    def _load_model(self):
        """Load and combine MuJoCo XML models."""
        
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, 'assets')
        
        # Load base quadruped model
        quadruped_path = os.path.join(assets_dir, 'quadruped.xml')
        parkour_path = os.path.join(assets_dir, 'parkour_course.xml')
        terrain_path = os.path.join(assets_dir, 'terrain_variations.xml')
        
        # Create combined model
        combined_xml = self._combine_models(quadruped_path, parkour_path, terrain_path)
        
        # Load the combined model
        self.model = mujoco.MjModel.from_xml_string(combined_xml)
        
        # Get important body and joint indices
        self._get_model_indices()
    
    def _combine_models(self, quadruped_path: str, parkour_path: str, terrain_path: str) -> str:
        """Combine multiple XML models into one."""
        
        # Parse the quadruped model as base
        with open(quadruped_path, 'r') as f:
            base_xml = f.read()
        
        base_tree = ET.fromstring(base_xml)
        
        # Parse parkour course
        with open(parkour_path, 'r') as f:
            parkour_xml = f.read()
        parkour_tree = ET.fromstring(parkour_xml)
        
        # Parse terrain variations
        with open(terrain_path, 'r') as f:
            terrain_xml = f.read()
        terrain_tree = ET.fromstring(terrain_xml)
        
        # Get worldbody from base
        base_worldbody = base_tree.find('worldbody')
        
        # Add parkour elements to base worldbody
        parkour_worldbody = parkour_tree.find('worldbody')
        if parkour_worldbody is not None:
            for child in parkour_worldbody:
                if child.tag != 'light':  # Skip duplicate lights
                    base_worldbody.append(child)
        
        # Add terrain elements to base worldbody
        terrain_worldbody = terrain_tree.find('worldbody')
        if terrain_worldbody is not None:
            for child in terrain_worldbody:
                if child.tag not in ['light', 'geom']:  # Skip lights and basic geoms
                    base_worldbody.append(child)
                elif child.tag == 'geom' and child.get('name') not in ['floor']:
                    base_worldbody.append(child)
        
        # Add actuators from other models
        base_actuator = base_tree.find('actuator')
        if base_actuator is None:
            base_actuator = ET.SubElement(base_tree, 'actuator')
        
        # Add parkour actuators
        parkour_actuator = parkour_tree.find('actuator')
        if parkour_actuator is not None:
            for motor in parkour_actuator:
                base_actuator.append(motor)
        
        # Add terrain actuators
        terrain_actuator = terrain_tree.find('actuator')
        if terrain_actuator is not None:
            for motor in terrain_actuator:
                base_actuator.append(motor)
        
        # Add assets from other models
        base_asset = base_tree.find('asset')
        if base_asset is None:
            base_asset = ET.SubElement(base_tree, 'asset')
        
        # Add parkour assets
        parkour_asset = parkour_tree.find('asset')
        if parkour_asset is not None:
            for asset in parkour_asset:
                if asset.get('name') not in [a.get('name') for a in base_asset]:
                    base_asset.append(asset)
        
        # Add terrain assets
        terrain_asset = terrain_tree.find('asset')
        if terrain_asset is not None:
            for asset in terrain_asset:
                if asset.get('name') not in [a.get('name') for a in base_asset]:
                    base_asset.append(asset)
        
        return ET.tostring(base_tree, encoding='unicode')
    
    def _get_model_indices(self):
        """Get important model indices for efficient access."""
        
        # Robot body indices
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        
        # Foot body indices
        self.foot_ids = {
            'fl': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'fl_foot'),
            'fr': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'fr_foot'),
            'bl': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bl_foot'),
            'br': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'br_foot')
        }
        
        # Joint indices (16 leg joints)
        self.joint_names = [
            'fl_hip_abduction', 'fl_hip_flexion', 'fl_knee', 'fl_ankle',
            'fr_hip_abduction', 'fr_hip_flexion', 'fr_knee', 'fr_ankle',
            'bl_hip_abduction', 'bl_hip_flexion', 'bl_knee', 'bl_ankle',
            'br_hip_abduction', 'br_hip_flexion', 'br_knee', 'br_ankle'
        ]
        
        self.joint_ids = []
        for name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.joint_ids.append(joint_id)
            except:
                print(f"Warning: Joint {name} not found")
        
        # Actuator indices
        self.actuator_names = [name + '_motor' for name in self.joint_names]
        self.actuator_ids = []
        for name in self.actuator_names:
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.actuator_ids.append(actuator_id)
            except:
                print(f"Warning: Actuator {name} not found")
        
        # Sensor indices for contact detection
        self.contact_sensor_names = ['fl_foot_contact', 'fr_foot_contact', 'bl_foot_contact', 'br_foot_contact']
        self.contact_sensor_ids = []
        for name in self.contact_sensor_names:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                self.contact_sensor_ids.append(sensor_id)
            except:
                print(f"Warning: Contact sensor {name} not found")
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        
        # Action space: 16 joint torques
        # Hip joints: ±80 Nm, Knee joints: ±60 Nm, Ankle joints: ±40 Nm
        action_limits = []
        for i, joint_name in enumerate(self.joint_names):
            if 'hip' in joint_name:
                action_limits.append(80.0)
            elif 'knee' in joint_name:
                action_limits.append(60.0)
            elif 'ankle' in joint_name:
                action_limits.append(40.0)
        
        self.action_space = spaces.Box(
            low=-np.array(action_limits),
            high=np.array(action_limits),
            dtype=np.float32
        )
        
        # Observation space: 95 elements total
        # - Joint positions (16) + Joint velocities (16) = 32
        # - Body orientation quaternion (4) + Body velocity (6) + Body position (3) = 13
        # - Foot contact states (4) + Foot positions (12) = 16
        # - Lidar readings (36) = 36
        # - Obstacle info (8) = 8
        # - Terrain info (2) + Distance to finish (1) = 3
        # Total: 32 + 13 + 16 + 36 + 8 + 3 = 108 -> Reduced to 95
        
        obs_low = np.full(95, -np.inf, dtype=np.float32)
        obs_high = np.full(95, np.inf, dtype=np.float32)
        
        # Joint positions have limits
        obs_low[0:16] = -np.pi
        obs_high[0:16] = np.pi
        
        # Joint velocities
        obs_low[16:32] = -20.0
        obs_high[16:32] = 20.0
        
        # Body orientation (quaternion)
        obs_low[32:36] = -1.0
        obs_high[32:36] = 1.0
        
        # Foot contact states (binary)
        obs_low[48:52] = 0.0
        obs_high[48:52] = 1.0
        
        # Lidar readings (distances)
        obs_low[64:88] = 0.0
        obs_high[64:88] = 10.0
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
    
    def _get_obstacle_positions(self) -> List[Tuple[float, str]]:
        """Get positions and types of obstacles in the course."""
        
        obstacles = [
            (8.0, 'stairs_small'),
            (16.0, 'gap_jump'),
            (24.0, 'balance_beam'),
            (30.0, 'ramp_up'),
            (36.0, 'narrow_passage'),
            (44.0, 'moving_platform'),
            (50.0, 'rough_terrain'),
            (58.0, 'stairs_large'),
            (72.0, 'ramp_down'),
            (78.0, 'final_gap'),
            (88.0, 'pendulum'),
            (92.0, 'collapsing_bridge')
        ]
        
        return obstacles
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment's random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        
        if seed is not None:
            self.seed(seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial robot position and orientation
        self.data.qpos[:] = self.initial_qpos.copy()
        self.data.qvel[:] = self.initial_qvel.copy()
        
        # Set robot position
        if self.torso_id >= 0:
            # Set position (first 3 elements of qpos for free joint)
            self.data.qpos[0:3] = self.start_pos
            # Set orientation (quaternion, next 4 elements)
            self.data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
        
        # Reset episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_position = self.start_pos.copy()
        self.max_forward_progress = 0.0
        self.checkpoints_reached = set()
        self.fall_count = 0
        self.stuck_counter = 0
        
        # Randomize dynamic obstacles
        self._randomize_obstacles()
        
        # Step simulation to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply actions to actuators
        if len(self.actuator_ids) >= len(action):
            self.data.ctrl[:len(action)] = action
        
        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Update dynamic obstacles
        self._update_dynamic_obstacles()
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        # Update tracking
        self.step_count += 1
        self.episode_reward += reward
        
        # Get info
        info = self._get_info()
        
        # Render if needed (like working MuJoCo experiments)
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        
        obs = np.zeros(95, dtype=np.float32)
        idx = 0
        
        # Joint positions (16)
        if len(self.joint_ids) >= 16:
            joint_positions = self.data.qpos[7:7+16]  # Skip free joint (7 DOF)
            obs[idx:idx+16] = joint_positions
        idx += 16
        
        # Joint velocities (16)
        if len(self.joint_ids) >= 16:
            joint_velocities = self.data.qvel[6:6+16]  # Skip free joint (6 DOF)
            obs[idx:idx+16] = joint_velocities
        idx += 16
        
        # Body orientation (quaternion, 4)
        if self.torso_id >= 0:
            body_quat = self.data.qpos[3:7]  # Quaternion from free joint
            obs[idx:idx+4] = body_quat
        idx += 4
        
        # Body linear velocity (3)
        if self.torso_id >= 0:
            body_vel = self.data.qvel[0:3]  # Linear velocity from free joint
            obs[idx:idx+3] = body_vel
        idx += 3
        
        # Body angular velocity (3)
        if self.torso_id >= 0:
            body_angvel = self.data.qvel[3:6]  # Angular velocity from free joint
            obs[idx:idx+3] = body_angvel
        idx += 3
        
        # Body position (3)
        if self.torso_id >= 0:
            body_pos = self.data.qpos[0:3]  # Position from free joint
            obs[idx:idx+3] = body_pos
        idx += 3
        
        # Foot contact states (4)
        foot_contacts = self._get_foot_contacts()
        obs[idx:idx+4] = foot_contacts
        idx += 4
        
        # Foot positions relative to body (12)
        foot_positions = self._get_foot_positions()
        obs[idx:idx+12] = foot_positions.flatten()
        idx += 12
        
        # Lidar readings (24, reduced from 36)
        lidar_readings = self._get_lidar_readings()
        obs[idx:idx+24] = lidar_readings
        idx += 24
        
        # Upcoming obstacle information (8)
        obstacle_info = self._get_obstacle_info()
        obs[idx:idx+8] = obstacle_info
        idx += 8
        
        # Terrain slope and friction (2)
        terrain_info = self._get_terrain_info()
        obs[idx:idx+2] = terrain_info
        idx += 2
        
        # Distance to finish (1)
        if idx < len(obs):
            distance_to_finish = self._get_distance_to_finish()
            obs[idx] = distance_to_finish
        
        return obs
    
    def _get_foot_contacts(self) -> np.ndarray:
        """Get foot contact states."""
        
        contacts = np.zeros(4, dtype=np.float32)
        
        # Check contacts for each foot
        for i, foot_id in enumerate(self.foot_ids.values()):
            if foot_id >= 0:
                # Check if foot is in contact with ground
                for j in range(self.data.ncon):
                    contact = self.data.contact[j]
                    if (contact.geom1 == foot_id or contact.geom2 == foot_id):
                        contacts[i] = 1.0
                        break
        
        return contacts
    
    def _get_foot_positions(self) -> np.ndarray:
        """Get foot positions relative to body."""
        
        positions = np.zeros((4, 3), dtype=np.float32)
        
        # Get body position
        if self.torso_id >= 0:
            body_pos = self.data.xpos[self.torso_id]
            
            # Get foot positions
            for i, foot_id in enumerate(self.foot_ids.values()):
                if foot_id >= 0:
                    foot_pos = self.data.xpos[foot_id]
                    positions[i] = foot_pos - body_pos
        
        return positions
    
    def _get_lidar_readings(self) -> np.ndarray:
        """Simulate lidar readings."""
        
        readings = np.full(24, 10.0, dtype=np.float32)  # Default max range
        
        if self.torso_id >= 0:
            body_pos = self.data.xpos[self.torso_id]
            body_mat = self.data.xmat[self.torso_id].reshape(3, 3)
            
            # Cast rays in 180° arc in front of robot (24 rays, 7.5° apart)
            for i in range(24):
                angle = -np.pi/2 + i * np.pi/23  # -90° to +90°
                
                # Ray direction in body frame
                ray_dir_body = np.array([np.cos(angle), np.sin(angle), 0])
                
                # Transform to world frame
                ray_dir_world = body_mat @ ray_dir_body
                
                # Cast ray
                ray_start = body_pos + np.array([0, 0, 0.1])  # Slightly above ground
                ray_end = ray_start + ray_dir_world * 10.0  # 10m max range
                
                # Simple collision detection (would need proper implementation)
                readings[i] = 10.0  # Placeholder
        
        return readings
    
    def _get_obstacle_info(self) -> np.ndarray:
        """Get information about upcoming obstacles."""
        
        info = np.zeros(8, dtype=np.float32)
        
        if self.torso_id >= 0:
            current_x = self.data.xpos[self.torso_id][0]
            
            # Find next two obstacles
            upcoming_obstacles = []
            for pos, obstacle_type in self.obstacle_positions:
                if pos > current_x:
                    upcoming_obstacles.append((pos, obstacle_type))
                if len(upcoming_obstacles) >= 2:
                    break
            
            # Encode obstacle information
            for i, (pos, obs_type) in enumerate(upcoming_obstacles):
                if i < 2:
                    base_idx = i * 4
                    info[base_idx] = pos - current_x  # Distance
                    info[base_idx + 1] = self._encode_obstacle_type(obs_type)  # Type
                    info[base_idx + 2] = self._get_obstacle_height(obs_type)  # Height
                    info[base_idx + 3] = self._get_obstacle_difficulty(obs_type)  # Difficulty
        
        return info
    
    def _encode_obstacle_type(self, obstacle_type: str) -> float:
        """Encode obstacle type as float."""
        
        type_map = {
            'stairs_small': 1.0,
            'gap_jump': 2.0,
            'balance_beam': 3.0,
            'ramp_up': 4.0,
            'narrow_passage': 5.0,
            'moving_platform': 6.0,
            'rough_terrain': 7.0,
            'stairs_large': 8.0,
            'ramp_down': 9.0,
            'final_gap': 10.0,
            'pendulum': 11.0,
            'collapsing_bridge': 12.0
        }
        
        return type_map.get(obstacle_type, 0.0)
    
    def _get_obstacle_height(self, obstacle_type: str) -> float:
        """Get obstacle height."""
        
        height_map = {
            'stairs_small': 0.225,
            'gap_jump': 0.2,
            'balance_beam': 0.5,
            'ramp_up': 0.6,
            'narrow_passage': 0.6,
            'moving_platform': 0.3,
            'rough_terrain': 0.08,
            'stairs_large': 0.4,
            'ramp_down': 0.25,
            'final_gap': 0.3,
            'pendulum': 0.0,
            'collapsing_bridge': 0.2
        }
        
        return height_map.get(obstacle_type, 0.0)
    
    def _get_obstacle_difficulty(self, obstacle_type: str) -> float:
        """Get obstacle difficulty (0-1 scale)."""
        
        difficulty_map = {
            'stairs_small': 0.3,
            'gap_jump': 0.6,
            'balance_beam': 0.8,
            'ramp_up': 0.4,
            'narrow_passage': 0.7,
            'moving_platform': 0.9,
            'rough_terrain': 0.5,
            'stairs_large': 0.6,
            'ramp_down': 0.4,
            'final_gap': 0.8,
            'pendulum': 1.0,
            'collapsing_bridge': 1.0
        }
        
        return difficulty_map.get(obstacle_type, 0.0)
    
    def _get_terrain_info(self) -> np.ndarray:
        """Get terrain information at current position."""
        
        info = np.zeros(2, dtype=np.float32)
        
        if self.torso_id >= 0:
            # Simplified terrain analysis
            current_pos = self.data.xpos[self.torso_id]
            
            # Estimate slope (placeholder)
            info[0] = 0.0  # Slope
            
            # Estimate friction (placeholder)
            info[1] = 0.8  # Friction coefficient
        
        return info
    
    def _get_distance_to_finish(self) -> float:
        """Get distance to finish line."""
        
        if self.torso_id >= 0:
            current_pos = self.data.xpos[self.torso_id]
            distance = np.linalg.norm(current_pos[:2] - self.finish_pos[:2])
            return distance
        
        return 100.0
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current step."""
        
        reward = 0.0
        
        if self.torso_id < 0:
            return -20.0  # Penalty for invalid state
        
        current_pos = self.data.xpos[self.torso_id]
        current_x = current_pos[0]
        
        # Time penalty (encourages fast completion)
        reward -= 20.0
        
        # Forward progress reward
        progress = current_x - self.last_position[0]
        if progress > 0:
            reward += progress * 500.0  # Scale progress reward
            self.max_forward_progress = max(self.max_forward_progress, current_x)
        elif progress < -0.1:  # Moving backward penalty
            reward -= 100.0
        
        # Checkpoint rewards
        for checkpoint_x in self.checkpoint_positions:
            if (checkpoint_x not in self.checkpoints_reached and 
                current_x >= checkpoint_x):
                self.checkpoints_reached.add(checkpoint_x)
                reward += 1000.0
        
        # Obstacle completion rewards
        for obs_x, obs_type in self.obstacle_positions:
            obstacle_key = f"{obs_x}_{obs_type}"
            if (obstacle_key not in self.checkpoints_reached and 
                current_x > obs_x + 2.0):  # Passed obstacle
                self.checkpoints_reached.add(obstacle_key)
                difficulty = self._get_obstacle_difficulty(obs_type)
                reward += 1000.0 + difficulty * 1000.0
        
        # Course completion reward
        if current_x >= self.finish_pos[0]:
            reward += 5000.0
        
        # Stability rewards
        body_quat = self.data.qpos[3:7]
        # Check if robot is upright (quaternion w component should be close to 1)
        uprightness = abs(body_quat[0])  # w component of quaternion
        if uprightness > 0.7:
            reward += 100.0
        
        # Foot contact reward (encourage proper gait)
        foot_contacts = self._get_foot_contacts()
        contact_count = np.sum(foot_contacts)
        if 1 <= contact_count <= 3:  # Good gait pattern
            reward += 200.0
        
        # Energy efficiency (penalize excessive joint motion)
        joint_effort = np.sum(np.abs(action))
        reward -= joint_effort * 0.1
        
        # Fall penalty
        if current_pos[2] < 0.2:  # Robot body too low
            reward -= 2000.0
            self.fall_count += 1
        
        # Collision penalty (simplified)
        if self.data.ncon > 8:  # Too many contacts (likely collision)
            reward -= 500.0
        
        # Stuck penalty
        if abs(progress) < 0.01:
            self.stuck_counter += 1
            if self.stuck_counter > 100:  # Stuck for 1 second
                reward -= 100.0
        else:
            self.stuck_counter = 0
        
        # Update last position
        self.last_position = current_pos.copy()
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        
        if self.torso_id < 0:
            return True
        
        current_pos = self.data.xpos[self.torso_id]
        
        # Course completed
        if current_pos[0] >= self.finish_pos[0]:
            return True
        
        # Robot fell over
        if current_pos[2] < 0.15:  # Body touching ground
            return True
        
        # Robot fell off course
        if abs(current_pos[1]) > self.course_width / 2:
            return True
        
        # Robot stuck for too long
        if self.stuck_counter > 1000:  # 10 seconds
            return True
        
        # Too many falls
        if self.fall_count > 3:
            return True
        
        return False
    
    def _randomize_obstacles(self):
        """Randomize dynamic obstacle positions and states."""
        
        # Randomize moving platform position
        try:
            platform_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'platform_slide')
            if platform_joint_id >= 0:
                self.data.qpos[platform_joint_id] = self.np_random.uniform(-1.5, 1.5)
        except:
            pass
        
        # Randomize pendulum position
        try:
            pendulum_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'pendulum_swing')
            if pendulum_joint_id >= 0:
                self.data.qpos[pendulum_joint_id] = self.np_random.uniform(-1.0, 1.0)
        except:
            pass
    
    def _update_dynamic_obstacles(self):
        """Update dynamic obstacles during simulation."""
        
        # Simple sinusoidal motion for moving platform
        try:
            platform_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'platform_motor')
            if platform_motor_id >= 0:
                t = self.step_count * self.dt
                self.data.ctrl[platform_motor_id] = 50.0 * np.sin(0.5 * t)
        except:
            pass
        
        # Pendulum motion
        try:
            pendulum_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'pendulum_motor')
            if pendulum_motor_id >= 0:
                t = self.step_count * self.dt
                self.data.ctrl[pendulum_motor_id] = 100.0 * np.sin(0.3 * t)
        except:
            pass
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        
        info = {
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'max_forward_progress': self.max_forward_progress,
            'checkpoints_reached': len(self.checkpoints_reached),
            'fall_count': self.fall_count,
            'course_completion': 0.0
        }
        
        if self.torso_id >= 0:
            current_x = self.data.xpos[self.torso_id][0]
            info['course_completion'] = min(1.0, max(0.0, (current_x - self.start_pos[0]) / (self.finish_pos[0] - self.start_pos[0])))
        
        return info
    
    def _init_viewer(self):
        """Initialize MuJoCo viewer for rendering."""
        
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except ImportError:
            print("MuJoCo viewer not available. Install with: pip install mujoco[viewer]")
            self.viewer = None
    
    def render(self):
        """Render the environment."""
        
        if self.render_mode == 'human':
            if self.viewer is None:
                self._init_viewer()
            
            if self.viewer is not None:
                self.viewer.sync()
        
        elif self.render_mode == 'rgb_array':
            # Render to RGB array (would need proper implementation)
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        """Close the environment."""
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Register environment
if __name__ == "__main__":
    # Test basic functionality
    env = QuadrupedParkourEnv(render_mode='human')
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
        
        if i % 10 == 0:
            print(f"Step {i}: Reward = {reward:.2f}, Progress = {info['course_completion']:.2%}")
    
    env.close()
