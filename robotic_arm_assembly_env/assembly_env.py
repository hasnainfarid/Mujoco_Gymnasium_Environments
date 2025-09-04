"""
Robotic Arm Assembly Environment
A MuJoCo-based environment for precision robotic assembly tasks
Author: Hasnain Fareed
Year: 2025
"""

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import os
from typing import Optional, Dict, Any, Tuple, List
import time
import warnings

class RoboticArmAssemblyEnv(gym.Env):
    """
    MuJoCo-based robotic arm assembly environment with precision manipulation tasks.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }
    
    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize the robotic arm assembly environment."""
        super().__init__()
        
        self.render_mode = render_mode
        self.config = config or {}
        
        # Environment parameters
        self.max_episode_steps = 150000  # 5 minutes at 500Hz
        self.control_frequency = 50  # Hz
        self.simulation_frequency = 500  # Hz
        self.skip_frames = self.simulation_frequency // self.control_frequency
        
        # Assembly task parameters
        self.assembly_tolerance = 0.002  # 2mm position tolerance
        self.force_threshold = 50.0  # Max force before damage
        self.gentle_force_threshold = 10.0  # Force for gentle handling
        
        # Component assembly order
        self.assembly_sequence = [
            'pcb', 'screw1', 'screw2', 'screw3', 'screw4',
            'cpu', 'battery', 'cable', 'cover'
        ]
        
        # Load MuJoCo model
        model_path = os.path.join(os.path.dirname(__file__), 'assets', 'complete_model.xml')
        if not os.path.exists(model_path):
            # Fallback to gripper model if complete doesn't exist
            model_path = os.path.join(os.path.dirname(__file__), 'assets', 'gripper.xml')
            if not os.path.exists(model_path):
                # Final fallback to robot arm model
                model_path = os.path.join(os.path.dirname(__file__), 'assets', 'robot_arm.xml')
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Initialize tracking variables
        self.step_count = 0
        self.assembly_progress = {}
        self.component_status = {}
        self.gripper_contact_body = None
        self.task_phase = 'idle'  # idle, pickup, transport, align, insert
        self.held_component = None
        self.cumulative_reward = 0
        
        # Component target positions (assembly locations)
        self.component_targets = {
            'pcb': np.array([0, 0, 0.74]),
            'cpu': np.array([0, 0, 0.76]),
            'screw1': np.array([-0.08, -0.06, 0.735]),
            'screw2': np.array([0.08, -0.06, 0.735]),
            'screw3': np.array([-0.08, 0.06, 0.735]),
            'screw4': np.array([0.08, 0.06, 0.735]),
            'battery': np.array([0.05, 0, 0.77]),
            'cable': np.array([-0.05, 0, 0.77]),
            'cover': np.array([0, 0, 0.79])
        }
        
        # Initialize viewer if needed
        self.viewer = None
        if self.render_mode == "human":
            self._init_viewer()
        
        # Reset environment
        self.reset()
    
    def _setup_spaces(self):
        """Define observation and action spaces."""
        # Observation space: 110 elements
        obs_low = np.full(110, -np.inf)
        obs_high = np.full(110, np.inf)
        
        # Joint position limits (7 joints)
        obs_low[0:7] = np.array([-3.14, -2.36, -2.97, -3.14, -2.09, -3.14, -3.14])
        obs_high[0:7] = np.array([3.14, 0.78, 2.97, 3.14, 2.09, 3.14, 3.14])
        
        # Joint velocity limits (7 joints)
        obs_low[7:14] = -5.0
        obs_high[7:14] = 5.0
        
        # Gripper state (opening and force)
        obs_low[14:16] = np.array([0, 0])
        obs_high[14:16] = np.array([0.1, 50])
        
        # End-effector pose (position + quaternion)
        obs_low[16:19] = np.array([-2, -2, 0])
        obs_high[16:19] = np.array([2, 2, 2])
        obs_low[19:23] = -1.0  # Quaternion
        obs_high[19:23] = 1.0
        
        # Component positions and orientations (8 components × 7)
        obs_low[23:79] = np.tile([-2, -2, 0, -1, -1, -1, -1], 8)
        obs_high[23:79] = np.tile([2, 2, 2, 1, 1, 1, 1], 8)
        
        # Component assembly status (8 binary)
        obs_low[79:87] = 0
        obs_high[79:87] = 1
        
        # Gripper contact and held object
        obs_low[87:89] = np.array([0, -1])
        obs_high[87:89] = np.array([1, 8])
        
        # Camera features (simplified)
        obs_low[89:114] = 0
        obs_high[89:114] = 1
        
        # Force/torque at wrist
        obs_low[104:110] = -100
        obs_high[104:110] = 100
        
        # Assembly progress and task phase
        obs_low[108] = 0
        obs_high[108] = 100
        obs_low[109] = 0
        obs_high[109] = 4
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: 9 dimensions (7 joint velocities + gripper opening + grip force)
        action_low = np.array([-2, -2, -2, -2, -2, -2, -2, 0, 0])
        action_high = np.array([2, 2, 2, 2, 2, 2, 2, 100, 50])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    
    def _init_viewer(self):
        """Initialize the MuJoCo viewer for visualization."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 120
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Initialize robot arm to home position
        if self.model.nq >= 7:
            self.data.qpos[0:7] = np.array([0, -0.5, 0.5, 0, 0.5, 0, 0])  # Home position
        
        # Reset components to their bins
        self._reset_components()
        
        # Reset tracking variables
        self.step_count = 0
        self.assembly_progress = {comp: False for comp in self.assembly_sequence}
        self.component_status = {comp: 'in_bin' for comp in self.assembly_sequence}
        self.gripper_contact_body = None
        self.task_phase = 'idle'
        self.held_component = None
        self.cumulative_reward = 0
        
        # Step simulation to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _reset_components(self):
        """Reset all components to their initial bin positions."""
        # Component initial positions in bins
        initial_positions = {
            'pcb': [-0.6, 0.3, 0.76],
            'cpu': [-0.6, 0, 0.76],
            'screw1': [-0.6, -0.3, 0.76],
            'screw2': [-0.58, -0.3, 0.76],
            'screw3': [-0.62, -0.3, 0.76],
            'screw4': [-0.6, -0.28, 0.76],
            'battery': [0.6, 0.3, 0.76],
            'cable': [0.6, -0.3, 0.76],
            'cover': [0.6, 0, 0.76]
        }
        
        # Set component positions if bodies exist
        for comp_name, pos in initial_positions.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, comp_name)
            if body_id >= 0:
                # Find the joint for this body (assuming freejoint)
                joint_adr = self.model.body_jntadr[body_id]
                if joint_adr >= 0:
                    qpos_adr = self.model.jnt_qposadr[joint_adr]
                    self.data.qpos[qpos_adr:qpos_adr+3] = pos
                    self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]  # Identity quaternion
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.step_count += 1
        
        # Apply control action
        self._apply_action(action)
        
        # Step physics simulation
        for _ in range(self.skip_frames):
            mujoco.mj_step(self.model, self.data)
        
        # Update task phase and component status
        self._update_task_state()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.cumulative_reward += reward
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Apply control action to the robot."""
        # Clip actions to valid ranges
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply joint velocities (first 7 actions)
        if self.model.nu >= 7:
            self.data.ctrl[0:7] = action[0:7]
        
        # Apply gripper control (action[7] is opening, action[8] is force)
        if self.model.nu >= 9:
            gripper_opening = action[7] / 1000.0  # Convert mm to meters
            self.data.ctrl[7] = gripper_opening  # Left finger
            self.data.ctrl[8] = gripper_opening  # Right finger
    
    def _update_task_state(self):
        """Update task phase and component status based on current state."""
        # Check gripper contacts
        gripper_contacts = self._get_gripper_contacts()
        
        if gripper_contacts:
            # Gripper is holding something
            if self.held_component is None:
                # Just picked up
                self.held_component = gripper_contacts[0]
                self.task_phase = 'pickup'
                self.component_status[self.held_component] = 'held'
            else:
                self.task_phase = 'transport'
        elif self.held_component:
            # Just released component
            comp_pos = self._get_component_position(self.held_component)
            target_pos = self.component_targets[self.held_component]
            
            # Check if placed correctly
            if np.linalg.norm(comp_pos - target_pos) < self.assembly_tolerance:
                self.assembly_progress[self.held_component] = True
                self.component_status[self.held_component] = 'assembled'
                self.task_phase = 'insert'
            else:
                self.component_status[self.held_component] = 'dropped'
                self.task_phase = 'idle'
            
            self.held_component = None
        else:
            self.task_phase = 'idle'
    
    def _get_gripper_contacts(self) -> List[str]:
        """Get list of components in contact with gripper."""
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            # Check if gripper pad is involved
            if geom1_name and geom2_name:
                if 'gripper' in geom1_name and 'pad' in geom1_name:
                    # Extract component name from geom2
                    for comp in self.assembly_sequence:
                        if comp in geom2_name:
                            contacts.append(comp)
                            break
                elif 'gripper' in geom2_name and 'pad' in geom2_name:
                    # Extract component name from geom1
                    for comp in self.assembly_sequence:
                        if comp in geom1_name:
                            contacts.append(comp)
                            break
        
        return list(set(contacts))  # Remove duplicates
    
    def _get_component_position(self, component: str) -> np.ndarray:
        """Get the position of a component."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, component)
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return np.zeros(3)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on current state and action."""
        reward = -10  # Base time penalty
        
        # Task completion rewards
        if self.task_phase == 'pickup' and self.held_component:
            reward += 1000  # Successful pickup
        
        # Assembly progress rewards
        for comp, assembled in self.assembly_progress.items():
            if assembled and self.component_status[comp] == 'assembled':
                if comp == 'pcb':
                    reward += 2000  # PCB placed correctly
                elif comp in ['screw1', 'screw2', 'screw3', 'screw4']:
                    reward += 500  # Screw inserted
                elif comp == 'cpu':
                    reward += 2000  # CPU installed
                elif comp == 'battery':
                    reward += 1000  # Battery connected
                elif comp == 'cable':
                    reward += 1000  # Cable connected
                elif comp == 'cover':
                    reward += 1000  # Cover attached
        
        # Precision rewards
        if self.held_component:
            comp_pos = self._get_component_position(self.held_component)
            target_pos = self.component_targets[self.held_component]
            distance = np.linalg.norm(comp_pos - target_pos)
            
            if distance < 0.05:  # Within 5cm
                reward += 300 * (1 - distance / 0.05)  # Closer is better
        
        # Force penalties
        max_force = self._get_max_contact_force()
        if max_force > self.force_threshold:
            reward -= 5000  # Excessive force penalty
        elif max_force < self.gentle_force_threshold:
            reward += 200  # Gentle handling bonus
        
        # Smooth motion reward
        joint_velocities = self.data.qvel[0:7] if self.model.nv >= 7 else np.zeros(7)
        smoothness = -np.sum(np.abs(joint_velocities)) * 10
        reward += smoothness
        
        # Component damage penalties
        for comp, status in self.component_status.items():
            if status == 'dropped':
                reward -= 2000
            elif status == 'damaged':
                reward -= 5000
        
        # Complete assembly bonus
        if all(self.assembly_progress.values()):
            reward += 10000
        
        return reward
    
    def _get_max_contact_force(self) -> float:
        """Get maximum contact force in the current state."""
        max_force = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Simple force approximation based on penetration depth
            force = abs(contact.dist) * 1000  # Scale factor for force
            max_force = max(max_force, force)
        return max_force
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Success: All components assembled
        if all(self.assembly_progress.values()):
            return True
        
        # Failure: Component damaged
        if any(status == 'damaged' for status in self.component_status.values()):
            return True
        
        # Failure: Robot collision or joint limits exceeded
        joint_pos = self.data.qpos[0:7] if self.model.nq >= 7 else np.zeros(7)
        joint_limits_low = np.array([-3.14, -2.36, -2.97, -3.14, -2.09, -3.14, -3.14])
        joint_limits_high = np.array([3.14, 0.78, 2.97, 3.14, 2.09, 3.14, 3.14])
        
        if np.any(joint_pos < joint_limits_low * 0.95) or np.any(joint_pos > joint_limits_high * 0.95):
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(110, dtype=np.float32)
        
        # Joint positions (7)
        if self.model.nq >= 7:
            obs[0:7] = self.data.qpos[0:7]
        
        # Joint velocities (7)
        if self.model.nv >= 7:
            obs[7:14] = self.data.qvel[0:7]
        
        # Gripper state (2)
        if self.model.nq >= 9:
            obs[14] = (self.data.qpos[7] + self.data.qpos[8]) / 2.0 * 1000  # Average opening in mm
        obs[15] = self._get_max_contact_force()  # Grip force
        
        # End-effector pose (7)
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if ee_site_id >= 0:
            obs[16:19] = self.data.site_xpos[ee_site_id]
            # Get orientation (simplified)
            obs[19:23] = [1, 0, 0, 0]  # Placeholder quaternion
        
        # Component positions and orientations (8 × 7 = 56)
        idx = 23
        for comp in self.assembly_sequence:
            comp_pos = self._get_component_position(comp)
            obs[idx:idx+3] = comp_pos
            obs[idx+3:idx+7] = [1, 0, 0, 0]  # Placeholder quaternion
            idx += 7
        
        # Component assembly status (8)
        for i, comp in enumerate(self.assembly_sequence):
            obs[79+i] = float(self.assembly_progress[comp])
        
        # Gripper contact and held object (2)
        obs[87] = float(self.held_component is not None)
        obs[88] = self.assembly_sequence.index(self.held_component) if self.held_component else -1
        
        # Camera features (25) - simplified
        obs[89:114] = 0.5  # Placeholder
        
        # Force/torque at wrist (6)
        obs[104:110] = 0  # Placeholder
        
        # Assembly progress score (1)
        obs[108] = sum(self.assembly_progress.values()) / len(self.assembly_progress) * 100
        
        # Task phase (1)
        phase_map = {'idle': 0, 'pickup': 1, 'transport': 2, 'align': 3, 'insert': 4}
        obs[109] = phase_map.get(self.task_phase, 0)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            'step_count': self.step_count,
            'assembly_progress': self.assembly_progress.copy(),
            'component_status': self.component_status.copy(),
            'task_phase': self.task_phase,
            'held_component': self.held_component,
            'cumulative_reward': self.cumulative_reward,
            'success': all(self.assembly_progress.values())
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # Return RGB array for recording
            return self._get_rgb_array()
    
    def _get_rgb_array(self) -> np.ndarray:
        """Get RGB array of current view."""
        # Simplified - would need proper camera setup
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
