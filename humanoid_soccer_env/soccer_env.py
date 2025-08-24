"""
Humanoid Soccer Environment - A MuJoCo-based 3D soccer simulation

This environment simulates a humanoid robot playing soccer on a realistic field
with physics-based ball interactions, opponent goalkeeper, and environmental factors.
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


class HumanoidSoccerEnv(gym.Env):
    """
    A MuJoCo-based humanoid soccer environment.
    
    The robot must learn to control a 25-DOF humanoid to play soccer,
    including walking, ball control, and scoring goals.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 50
    }
    
    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        """Initialize the humanoid soccer environment."""
        super().__init__()
        
        # Environment parameters
        self.dt = 0.02  # 50 Hz simulation
        self.max_episode_steps = 5000  # 100 seconds at 50 Hz
        self.current_step = 0
        
        # Field dimensions (meters)
        self.field_length = 50.0  # 25m radius from center
        self.field_width = 30.0   # 15m radius from center
        self.goal_width = 7.32
        self.goal_height = 2.44
        
        # Ball properties
        self.ball_radius = 0.11
        self.ball_mass = 0.43
        
        # Robot properties
        self.robot_height = 1.2
        self.num_joints = 25  # Will be updated after model loading
        
        # Reward parameters
        self.goal_reward = 10000.0
        self.ball_contact_reward = 1000.0
        self.approach_ball_reward = 500.0
        self.upright_reward = 200.0
        self.forward_progress_reward = 100.0
        self.energy_penalty = -0.1
        self.fall_penalty = -1000.0
        
        # Wind and environmental factors
        self.wind_strength = 0.0
        self.wind_direction = np.array([0.0, 0.0])
        self.field_friction_variation = 0.1
        
        # Opponent goalkeeper parameters
        self.goalkeeper_speed = 2.0
        self.goalkeeper_reaction_time = 0.1
        
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
        
        # Initialize viewer for rendering (like working MuJoCo experiments)
        self.viewer = None
        if self.render_mode == "human":
            self._init_viewer()
        
        # Episode tracking
        self.episode_stats = {
            'goals_scored': 0,
            'ball_contacts': 0,
            'distance_traveled': 0.0,
            'time_upright': 0.0,
            'max_ball_speed': 0.0
        }
        
        # Previous state for reward calculation
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        self.prev_robot_quat = None
        self.ball_contact_history = []
        
        # Goal detection
        self.goal_scored = False
        self.ball_in_goal_zone = False
        
        # Seed for reproducibility
        self.np_random = None
        self.seed()
    
    def _load_xml_models(self):
        """Load and combine all XML model files into a single model."""
        
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, 'assets')
        
        # Load individual XML files
        humanoid_path = os.path.join(assets_dir, 'humanoid.xml')
        field_path = os.path.join(assets_dir, 'soccer_field.xml')
        ball_path = os.path.join(assets_dir, 'ball.xml')
        
        # Parse XML files
        humanoid_tree = ET.parse(humanoid_path)
        field_tree = ET.parse(field_path)
        ball_tree = ET.parse(ball_path)
        
        # Create combined model
        root = ET.Element('mujoco', model='humanoid_soccer')
        root.set('model', 'humanoid_soccer')
        
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
        motor_default.set('ctrlrange', '-150.0 150.0')
        
        # Combine assets
        assets = ET.SubElement(root, 'asset')
        for child in field_tree.getroot().find('asset'):
            assets.append(child)
        for child in ball_tree.getroot().find('asset'):
            assets.append(child)
        for child in humanoid_tree.getroot().find('asset'):
            assets.append(child)
        
        # Combine worldbody
        worldbody = ET.SubElement(root, 'worldbody')
        
        # Add field elements
        for child in field_tree.getroot().find('worldbody'):
            worldbody.append(child)
        
        # Add ball elements
        for child in ball_tree.getroot().find('worldbody'):
            worldbody.append(child)
        
        # Add humanoid elements
        for child in humanoid_tree.getroot().find('worldbody'):
            worldbody.append(child)
        
        # Combine sensors
        sensors = ET.SubElement(root, 'sensor')
        for tree in [field_tree, ball_tree, humanoid_tree]:
            sensor_elem = tree.getroot().find('sensor')
            if sensor_elem is not None:
                for child in sensor_elem:
                    sensors.append(child)
        
        # Combine actuators
        actuators = ET.SubElement(root, 'actuator')
        actuator_elem = humanoid_tree.getroot().find('actuator')
        if actuator_elem is not None:
            for child in actuator_elem:
                actuators.append(child)
        
        # Add contact settings from ball.xml
        contact = ET.SubElement(root, 'contact')
        ball_contact = ball_tree.getroot().find('contact')
        if ball_contact is not None:
            for child in ball_contact:
                contact.append(child)
        
        # Convert to string
        self.xml_string = ET.tostring(root, encoding='unicode')
    
    def _get_model_indices(self):
        """Get indices for important model elements."""
        
        # Joint indices
        self.joint_names = [
            'abdomen_y', 'abdomen_z', 'abdomen_x', 'neck_x', 'neck_y',
            'right_shoulder1', 'right_shoulder2', 'right_elbow', 'right_wrist_y', 'right_wrist_x', 'right_wrist_z',
            'left_shoulder1', 'left_shoulder2', 'left_elbow', 'left_wrist_y', 'left_wrist_x', 'left_wrist_z',
            'right_hip_x', 'right_hip_z', 'right_hip_y', 'right_knee', 'right_ankle_y', 'right_ankle_x',
            'left_hip_x', 'left_hip_z', 'left_hip_y', 'left_knee', 'left_ankle_y', 'left_ankle_x'
        ]
        
        self.joint_indices = []
        for name in self.joint_names:
            try:
                idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.joint_indices.append(idx)
            except:
                print(f"Warning: Joint {name} not found")
        
        # Body indices
        try:
            self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
            self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
            self.goalkeeper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'opponent_goalkeeper')
        except:
            print("Warning: Some bodies not found")
        
        # Geom indices for contact detection
        try:
            self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')
            self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'right_foot')
            self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'left_foot')
            self.goal_zone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'east_goal_zone')
        except:
            print("Warning: Some geoms not found")
        
        # Site indices
        try:
            self.ball_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ball_center')
        except:
            print("Warning: Ball site not found")
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        
        # Action space: torque control for all joints
        self.action_space = spaces.Box(
            low=-150.0,
            high=150.0,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # Observation space: variable-dimensional state vector based on actual joints
        num_obs_joints = min(self.num_joints, 25)  # Cap at 25 for consistency
        obs_low = np.array([
            # Joint positions - normalized to [-1, 1]
            *[-1.0] * num_obs_joints,
            # Joint velocities - normalized to [-1, 1]  
            *[-1.0] * num_obs_joints,
            # Torso orientation quaternion (4) - normalized
            *[-1.0] * 4,
            # Torso linear velocity (3) - normalized to [-1, 1]
            *[-1.0] * 3,
            # Torso angular velocity (3) - normalized to [-1, 1]
            *[-1.0] * 3,
            # Ball position relative to robot (3) - normalized
            *[-1.0] * 3,
            # Ball velocity (3) - normalized
            *[-1.0] * 3,
            # Goal position relative to robot (3) - normalized
            *[-1.0] * 3,
            # Contact forces (4) - normalized
            *[-1.0] * 4,
            # Robot COM position (3) - normalized
            *[-1.0] * 3,
            # Time remaining (1) - normalized to [0, 1]
            0.0,
            # Distance to ball (1) - normalized
            0.0,
            # Opponent goalkeeper position (2) - normalized
            *[-1.0] * 2
        ], dtype=np.float32)
        
        obs_high = np.array([
            # Joint positions
            *[1.0] * num_obs_joints,
            # Joint velocities
            *[1.0] * num_obs_joints,
            # Torso orientation quaternion (4)
            *[1.0] * 4,
            # Torso linear velocity (3)
            *[1.0] * 3,
            # Torso angular velocity (3)
            *[1.0] * 3,
            # Ball position relative to robot (3)
            *[1.0] * 3,
            # Ball velocity (3)
            *[1.0] * 3,
            # Goal position relative to robot (3)
            *[1.0] * 3,
            # Contact forces (4)
            *[1.0] * 4,
            # Robot COM position (3)
            *[1.0] * 3,
            # Time remaining (1)
            1.0,
            # Distance to ball (1)
            1.0,
            # Opponent goalkeeper position (2)
            *[1.0] * 2
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
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
        self.goal_scored = False
        self.ball_in_goal_zone = False
        self.ball_contact_history = []
        
        # Reset episode stats
        self.episode_stats = {
            'goals_scored': 0,
            'ball_contacts': 0,
            'distance_traveled': 0.0,
            'time_upright': 0.0,
            'max_ball_speed': 0.0
        }
        
        # Randomize initial positions
        self._randomize_initial_state()
        
        # Update environmental factors
        self._update_environmental_factors()
        
        # Step simulation to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Get initial observation
        obs = self._get_observation()
        
        # Store previous state
        self.prev_ball_pos = self._get_ball_position().copy()
        self.prev_robot_pos = self._get_robot_position().copy()
        self.prev_robot_quat = self._get_robot_orientation().copy()
        
        info = {
            'episode_stats': self.episode_stats.copy(),
            'ball_position': self._get_ball_position(),
            'robot_position': self._get_robot_position(),
            'goal_distance': self._get_goal_distance()
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply actions to actuators
        self.data.ctrl[:] = action
        
        # Update opponent goalkeeper
        self._update_goalkeeper()
        
        # Apply environmental effects
        self._apply_environmental_effects()
        
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
        
        # Update episode stats
        self._update_episode_stats()
        
        # Prepare info
        info = {
            'episode_stats': self.episode_stats.copy(),
            'ball_position': self._get_ball_position(),
            'robot_position': self._get_robot_position(),
            'goal_distance': self._get_goal_distance(),
            'ball_contact': self._check_ball_contact(),
            'robot_upright': self._is_robot_upright(),
            'goal_scored': self.goal_scored
        }
        
        # Update previous state
        self.prev_ball_pos = self._get_ball_position().copy()
        self.prev_robot_pos = self._get_robot_position().copy()
        self.prev_robot_quat = self._get_robot_orientation().copy()
        
        # Render if needed (like working MuJoCo experiments)
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _randomize_initial_state(self):
        """Randomize initial positions of robot and ball."""
        
        # Robot starting position (defensive half)
        robot_x = self.np_random.uniform(-15.0, -5.0)
        robot_y = self.np_random.uniform(-10.0, 10.0)
        robot_z = 1.4
        
        # Set robot position
        self.data.qpos[self.model.jnt_qposadr[0]:self.model.jnt_qposadr[0]+3] = [robot_x, robot_y, robot_z]
        
        # Robot orientation (facing goal)
        angle = self.np_random.uniform(-0.5, 0.5)  # Small random rotation
        quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        self.data.qpos[self.model.jnt_qposadr[0]+3:self.model.jnt_qposadr[0]+7] = quat
        
        # Ball starting position - always in front of player
        ball_x = robot_x + 2.0  # 2 meters in front of player
        ball_y = robot_y  # Same Y position as player (centered)
        ball_z = 0.15  # Updated to match new ball size
        
        # Set ball position
        ball_qpos_start = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_joint')]
        self.data.qpos[ball_qpos_start:ball_qpos_start+3] = [ball_x, ball_y, ball_z]
        
        # Randomize joint positions slightly
        for i, joint_idx in enumerate(self.joint_indices):
            if joint_idx < len(self.data.qpos):
                joint_range = self.model.jnt_range[joint_idx]
                if joint_range[0] < joint_range[1]:  # Joint has limits
                    mid_pos = (joint_range[0] + joint_range[1]) / 2
                    noise = self.np_random.uniform(-0.1, 0.1)
                    self.data.qpos[self.model.jnt_qposadr[joint_idx]] = np.clip(
                        mid_pos + noise, joint_range[0], joint_range[1]
                    )
        
        # Goalkeeper position
        goalkeeper_y = self.np_random.uniform(-2.0, 2.0)
        goalkeeper_joint_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'goalkeeper_y')
        self.data.qpos[self.model.jnt_qposadr[goalkeeper_joint_idx]] = goalkeeper_y
    
    def _update_environmental_factors(self):
        """Update wind and field conditions."""
        
        # Update wind
        self.wind_strength = self.np_random.uniform(0.0, 2.0)
        wind_angle = self.np_random.uniform(0, 2 * np.pi)
        self.wind_direction = np.array([np.cos(wind_angle), np.sin(wind_angle)])
        
        # Update field friction variation
        self.field_friction_variation = self.np_random.uniform(0.05, 0.15)
    
    def _update_goalkeeper(self):
        """Update opponent goalkeeper behavior."""
        
        # Simple PID controller to track ball
        ball_pos = self._get_ball_position()
        goalkeeper_pos = self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'goalkeeper_y')]]
        
        # Only react if ball is in opponent's half
        if ball_pos[0] < -10.0:
            target_y = np.clip(ball_pos[1], -3.0, 3.0)  # Stay within goal area
            error = target_y - goalkeeper_pos
            
            # Simple proportional control
            control_force = 50.0 * error
            control_force = np.clip(control_force, -100.0, 100.0)
            
            # Apply force to goalkeeper
            goalkeeper_joint_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'goalkeeper_y')
            self.data.qfrc_applied[goalkeeper_joint_idx] = control_force
    
    def _apply_environmental_effects(self):
        """Apply wind and other environmental effects."""
        
        # Apply wind to ball
        ball_pos = self._get_ball_position()
        if ball_pos[2] > 0.5:  # Ball is in air
            wind_force = self.wind_strength * self.wind_direction * 0.1
            ball_body_id = self.ball_id
            self.data.xfrc_applied[ball_body_id, :2] += wind_force
        
        # Apply field friction variation
        # This would modify contact parameters dynamically in a more advanced implementation
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        
        obs = []
        num_obs_joints = min(self.num_joints, 25)  # Cap at 25 for consistency
        
        # Joint positions - normalized to [-1, 1]
        joint_positions = []
        for i in range(num_obs_joints):
            if i < len(self.joint_indices) and self.joint_indices[i] < len(self.data.qpos):
                joint_idx = self.joint_indices[i]
                pos = self.data.qpos[self.model.jnt_qposadr[joint_idx]]
                joint_range = self.model.jnt_range[joint_idx]
                if joint_range[0] < joint_range[1]:
                    # Normalize to [-1, 1]
                    normalized_pos = 2 * (pos - joint_range[0]) / (joint_range[1] - joint_range[0]) - 1
                    joint_positions.append(np.clip(normalized_pos, -1.0, 1.0))
                else:
                    joint_positions.append(0.0)
            else:
                joint_positions.append(0.0)
        obs.extend(joint_positions)
        
        # Joint velocities - normalized to [-1, 1]
        joint_velocities = []
        for i in range(num_obs_joints):
            if i < len(self.joint_indices) and self.joint_indices[i] < len(self.data.qvel):
                joint_idx = self.joint_indices[i]
                vel = self.data.qvel[self.model.jnt_dofadr[joint_idx]]
                # Normalize velocity (assume max vel = 10 rad/s)
                normalized_vel = np.clip(vel / 10.0, -1.0, 1.0)
                joint_velocities.append(normalized_vel)
            else:
                joint_velocities.append(0.0)
        obs.extend(joint_velocities)
        
        # Torso orientation quaternion (4)
        torso_quat = self._get_robot_orientation()
        obs.extend(torso_quat)
        
        # Torso linear velocity (3) - normalized
        torso_vel = self.data.qvel[:3]  # First 3 DOF are translation
        normalized_torso_vel = np.clip(torso_vel / 5.0, -1.0, 1.0)  # Assume max vel = 5 m/s
        obs.extend(normalized_torso_vel)
        
        # Torso angular velocity (3) - normalized
        torso_angvel = self.data.qvel[3:6]  # Next 3 DOF are rotation
        normalized_torso_angvel = np.clip(torso_angvel / 10.0, -1.0, 1.0)  # Assume max angvel = 10 rad/s
        obs.extend(normalized_torso_angvel)
        
        # Ball position relative to robot (3) - normalized
        robot_pos = self._get_robot_position()
        ball_pos = self._get_ball_position()
        rel_ball_pos = ball_pos - robot_pos
        normalized_rel_ball_pos = np.clip(rel_ball_pos / 30.0, -1.0, 1.0)  # Normalize by field size
        obs.extend(normalized_rel_ball_pos)
        
        # Ball velocity (3) - normalized
        ball_vel = self._get_ball_velocity()
        normalized_ball_vel = np.clip(ball_vel / 20.0, -1.0, 1.0)  # Assume max ball vel = 20 m/s
        obs.extend(normalized_ball_vel)
        
        # Goal position relative to robot (3) - normalized
        goal_pos = np.array([24.5, 0.0, 1.22])  # East goal center
        rel_goal_pos = goal_pos - robot_pos
        normalized_rel_goal_pos = np.clip(rel_goal_pos / 30.0, -1.0, 1.0)
        obs.extend(normalized_rel_goal_pos)
        
        # Contact forces on feet (4) - normalized
        contact_forces = self._get_foot_contact_forces()
        normalized_contact_forces = np.clip(contact_forces / 1000.0, -1.0, 1.0)  # Normalize by 1000N
        obs.extend(normalized_contact_forces)
        
        # Robot center of mass position (3) - normalized
        com_pos = self._get_robot_com_position()
        normalized_com_pos = np.clip(com_pos / 30.0, -1.0, 1.0)
        obs.extend(normalized_com_pos)
        
        # Time remaining in episode (1) - normalized to [0, 1]
        time_remaining = 1.0 - (self.current_step / self.max_episode_steps)
        obs.append(time_remaining)
        
        # Distance to ball (1) - normalized
        ball_distance = np.linalg.norm(rel_ball_pos)
        normalized_ball_distance = np.clip(ball_distance / 50.0, 0.0, 1.0)  # Max distance = 50m
        obs.append(normalized_ball_distance)
        
        # Opponent goalkeeper position (2) - normalized
        goalkeeper_pos = self._get_goalkeeper_position()
        normalized_goalkeeper_pos = np.clip(goalkeeper_pos[:2] / 15.0, -1.0, 1.0)  # Normalize by field width
        obs.extend(normalized_goalkeeper_pos)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current step."""
        
        reward = 0.0
        
        # Goal scoring reward
        if self._check_goal_scored():
            reward += self.goal_reward
            self.goal_scored = True
            self.episode_stats['goals_scored'] += 1
        
        # Ball contact reward
        if self._check_ball_contact():
            reward += self.ball_contact_reward
            self.episode_stats['ball_contacts'] += 1
        
        # Approach ball reward
        ball_pos = self._get_ball_position()
        robot_pos = self._get_robot_position()
        current_ball_distance = np.linalg.norm(ball_pos - robot_pos)
        
        if self.prev_ball_pos is not None and self.prev_robot_pos is not None:
            prev_ball_distance = np.linalg.norm(self.prev_ball_pos - self.prev_robot_pos)
            if current_ball_distance < prev_ball_distance and current_ball_distance > 2.0:
                reward += self.approach_ball_reward * (prev_ball_distance - current_ball_distance)
        
        # Upright posture reward
        if self._is_robot_upright():
            reward += self.upright_reward
            self.episode_stats['time_upright'] += self.dt
        
        # Forward progress reward (toward goal)
        if self.prev_robot_pos is not None:
            goal_pos = np.array([24.5, 0.0, 0.0])
            prev_goal_distance = np.linalg.norm(self.prev_robot_pos - goal_pos)
            current_goal_distance = np.linalg.norm(robot_pos - goal_pos)
            
            if current_goal_distance < prev_goal_distance:
                reward += self.forward_progress_reward * (prev_goal_distance - current_goal_distance)
        
        # Energy penalty (encourage efficient movement)
        energy_cost = np.sum(np.square(action))
        reward += self.energy_penalty * energy_cost
        
        # Fall penalty
        if not self._is_robot_upright():
            reward += self.fall_penalty
        
        # Ball progress toward goal reward
        if self.prev_ball_pos is not None:
            goal_pos = np.array([24.5, 0.0, 0.0])
            prev_ball_goal_distance = np.linalg.norm(self.prev_ball_pos - goal_pos)
            current_ball_goal_distance = np.linalg.norm(ball_pos - goal_pos)
            
            if current_ball_goal_distance < prev_ball_goal_distance:
                reward += 300.0 * (prev_ball_goal_distance - current_ball_goal_distance)
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        
        # Goal scored
        if self.goal_scored:
            return True
        
        # Robot fell and can't get up
        if not self._is_robot_upright() and self.current_step > 100:
            # Check if robot has been down for too long
            return True
        
        # Ball went out of bounds significantly
        ball_pos = self._get_ball_position()
        if (abs(ball_pos[0]) > 30.0 or abs(ball_pos[1]) > 20.0 or 
            ball_pos[2] < -1.0 or ball_pos[2] > 10.0):
            return True
        
        # Robot went out of bounds
        robot_pos = self._get_robot_position()
        if (abs(robot_pos[0]) > 30.0 or abs(robot_pos[1]) > 20.0 or 
            robot_pos[2] < 0.0 or robot_pos[2] > 5.0):
            return True
        
        return False
    
    def _update_episode_stats(self):
        """Update episode statistics."""
        
        # Distance traveled
        if self.prev_robot_pos is not None:
            robot_pos = self._get_robot_position()
            distance = np.linalg.norm(robot_pos - self.prev_robot_pos)
            self.episode_stats['distance_traveled'] += distance
        
        # Max ball speed
        ball_vel = self._get_ball_velocity()
        ball_speed = np.linalg.norm(ball_vel)
        self.episode_stats['max_ball_speed'] = max(self.episode_stats['max_ball_speed'], ball_speed)
    
    # Helper methods for getting state information
    def _get_robot_position(self) -> np.ndarray:
        """Get robot torso position."""
        return self.data.xpos[self.torso_id].copy()
    
    def _get_robot_orientation(self) -> np.ndarray:
        """Get robot torso orientation quaternion."""
        return self.data.xquat[self.torso_id].copy()
    
    def _get_robot_com_position(self) -> np.ndarray:
        """Get robot center of mass position."""
        return self.data.subtree_com[self.torso_id].copy()
    
    def _get_ball_position(self) -> np.ndarray:
        """Get ball position."""
        return self.data.xpos[self.ball_id].copy()
    
    def _get_ball_velocity(self) -> np.ndarray:
        """Get ball velocity."""
        ball_joint_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_joint')
        ball_dof_start = self.model.jnt_dofadr[ball_joint_idx]
        return self.data.qvel[ball_dof_start:ball_dof_start+3].copy()
    
    def _get_goalkeeper_position(self) -> np.ndarray:
        """Get goalkeeper position."""
        return self.data.xpos[self.goalkeeper_id].copy()
    
    def _get_goal_distance(self) -> float:
        """Get distance to goal."""
        robot_pos = self._get_robot_position()
        goal_pos = np.array([24.5, 0.0, 0.0])
        return np.linalg.norm(robot_pos - goal_pos)
    
    def _get_foot_contact_forces(self) -> np.ndarray:
        """Get contact forces on feet."""
        forces = np.zeros(4)
        
        # Check contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Right foot contact
            if ((contact.geom1 == self.right_foot_id and contact.geom2 == 0) or  # 0 is ground
                (contact.geom2 == self.right_foot_id and contact.geom1 == 0)):
                forces[0] = contact.dist  # Normal force approximation
                forces[1] = np.linalg.norm(contact.friction[:2])  # Tangential force
            
            # Left foot contact
            if ((contact.geom1 == self.left_foot_id and contact.geom2 == 0) or
                (contact.geom2 == self.left_foot_id and contact.geom1 == 0)):
                forces[2] = contact.dist  # Normal force approximation
                forces[3] = np.linalg.norm(contact.friction[:2])  # Tangential force
        
        return forces
    
    def _check_ball_contact(self) -> bool:
        """Check if robot is in contact with ball."""
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Check if ball is in contact with any robot part
            if (contact.geom1 == self.ball_geom_id or contact.geom2 == self.ball_geom_id):
                # Check if other geom belongs to robot
                other_geom = contact.geom2 if contact.geom1 == self.ball_geom_id else contact.geom1
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                
                if geom_name and any(part in geom_name for part in 
                                   ['foot', 'shin', 'thigh', 'torso', 'head', 'hand', 'arm']):
                    return True
        
        return False
    
    def _check_goal_scored(self) -> bool:
        """Check if ball crossed the goal line."""
        
        ball_pos = self._get_ball_position()
        
        # Check if ball is in goal area
        if (ball_pos[0] > 24.0 and  # Past goal line
            abs(ball_pos[1]) < 3.66 and  # Within goal width
            ball_pos[2] < 2.44):  # Below crossbar
            return True
        
        return False
    
    def _is_robot_upright(self) -> bool:
        """Check if robot is upright."""
        
        # Get robot orientation
        quat = self._get_robot_orientation()
        
        # Convert quaternion to rotation matrix
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, quat.astype(np.float64))
        rot_mat = rot_mat.reshape(3, 3)
        
        # Check if z-axis is pointing up (dot product with [0,0,1])
        up_vector = rot_mat[:, 2]  # Third column is z-axis
        upright_threshold = 0.7  # cos(45 degrees) approximately
        
        return up_vector[2] > upright_threshold
    
    def _init_viewer(self):
        """Initialize MuJoCo viewer (like working experiments)."""
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("🎮 MuJoCo viewer opened successfully!")
        except Exception as e:
            print(f"⚠️ Could not open viewer: {e}")
            self.viewer = None
    
    def render(self):
        """Render the environment (like working MuJoCo experiments)."""
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
            id='HumanoidSoccer-v0',
            entry_point='humanoid_soccer_env.soccer_env:HumanoidSoccerEnv',
            max_episode_steps=5000,
            reward_threshold=8000.0,
        )
    except gym.error.Error:
        # Environment already registered
        pass


if __name__ == "__main__":
    # Test the environment
    register_env()
    env = HumanoidSoccerEnv(render_mode='human')
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i}: Reward = {reward:.2f}, Ball contact = {info['ball_contact']}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            obs, info = env.reset()
    
    env.close()
