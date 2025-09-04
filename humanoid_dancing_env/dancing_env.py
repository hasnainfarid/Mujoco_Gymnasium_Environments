"""
Humanoid Dancing Environment - A MuJoCo-based rhythmic dance simulation

This environment simulates a humanoid robot learning to perform various dance moves,
follow rhythm patterns, maintain balance, and execute choreographed sequences.

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
from typing import Optional, Tuple, Dict, Any, List
import math


class HumanoidDancingEnv(gym.Env):
    """
    A MuJoCo-based humanoid dancing environment.
    
    The robot must learn to control a humanoid to perform dance moves,
    maintain rhythm, balance, and execute choreographed sequences with style.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 60  # Higher FPS for smoother dance visualization
    }
    
    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        """Initialize the humanoid dancing environment."""
        super().__init__()
        
        # Environment parameters
        self.dt = 0.01667  # 60 Hz simulation for smooth dance moves
        self.max_episode_steps = 3600  # 60 seconds at 60 Hz (full song duration)
        self.current_step = 0
        
        # Dance floor dimensions (meters)
        self.floor_radius = 10.0  # Circular dance floor
        self.stage_height = 0.5
        
        # Music and rhythm parameters
        self.bpm = 120  # Beats per minute
        self.beat_interval = 60.0 / self.bpm  # Seconds between beats
        self.time_since_last_beat = 0.0
        self.beat_count = 0
        self.current_measure = 0  # 4 beats per measure
        
        # Dance move categories
        self.dance_moves = {
            'basic_step': {'difficulty': 1, 'energy': 0.5, 'style_points': 10},
            'spin': {'difficulty': 2, 'energy': 1.0, 'style_points': 20},
            'jump': {'difficulty': 2, 'energy': 1.5, 'style_points': 25},
            'moonwalk': {'difficulty': 3, 'energy': 0.8, 'style_points': 40},
            'robot_wave': {'difficulty': 2, 'energy': 0.6, 'style_points': 30},
            'freeze': {'difficulty': 1, 'energy': 0.2, 'style_points': 15},
            'hip_hop_bounce': {'difficulty': 2, 'energy': 0.7, 'style_points': 25},
            'breakdance_toprock': {'difficulty': 3, 'energy': 1.2, 'style_points': 35},
            'salsa_basic': {'difficulty': 2, 'energy': 0.8, 'style_points': 28},
            'ballet_pirouette': {'difficulty': 4, 'energy': 1.0, 'style_points': 50}
        }
        
        # Current dance sequence
        self.dance_sequence = []
        self.current_move_idx = 0
        self.move_start_time = 0.0
        self.move_duration = 2.0  # Default move duration in seconds
        
        # Robot properties
        self.robot_height = 1.8  # Taller for better dance presence
        self.num_joints = 25  # Will be updated after model loading
        
        # Reward parameters
        self.rhythm_reward = 100.0
        self.move_completion_reward = 200.0
        self.style_bonus = 50.0
        self.balance_reward = 30.0
        self.smoothness_reward = 20.0
        self.energy_penalty = -0.05
        self.fall_penalty = -500.0
        self.off_beat_penalty = -50.0
        self.creativity_bonus = 100.0
        
        # Performance metrics
        self.performance_score = 0.0
        self.combo_multiplier = 1.0
        self.perfect_moves = 0
        self.good_moves = 0
        self.missed_beats = 0
        
        # Visual effects parameters (for future visualization)
        self.spotlight_position = np.array([0.0, 0.0, 5.0])
        self.spotlight_color = np.array([1.0, 1.0, 1.0])
        self.disco_ball_rotation = 0.0
        self.stage_lights = []
        
        # Crowd reaction simulation
        self.crowd_excitement = 0.5  # 0 to 1
        self.applause_level = 0.0
        
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
        
        # Initialize viewer for rendering (following MuJoCo visualization fix)
        self.viewer = None
        if self.render_mode == "human":
            self._init_viewer()
        
        # Episode tracking
        self.episode_stats = {
            'total_score': 0,
            'perfect_moves': 0,
            'good_moves': 0,
            'missed_beats': 0,
            'longest_combo': 0,
            'energy_used': 0.0,
            'time_on_beat': 0.0,
            'creativity_score': 0.0,
            'crowd_rating': 0.0
        }
        
        # Previous state for reward calculation
        self.prev_joint_pos = None
        self.prev_joint_vel = None
        self.prev_robot_pos = None
        self.prev_robot_quat = None
        self.move_history = []
        
        # Seed for reproducibility
        self.np_random = None
        self.seed()
    
    def _load_xml_models(self):
        """Load and combine all XML model files into a single model."""
        
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, 'assets')
        
        # For now, create a complete model programmatically
        # since XML files will be created separately
        root = ET.Element('mujoco', model='humanoid_dancing')
        
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
        option.set('integrator', 'RK4')  # Better for smooth movements
        
        # Add size settings
        size = ET.SubElement(root, 'size')
        size.set('nconmax', '100')
        size.set('njmax', '500')
        
        # Add defaults
        default = ET.SubElement(root, 'default')
        
        joint_default = ET.SubElement(default, 'joint')
        joint_default.set('armature', '0.5')
        joint_default.set('damping', '5')
        joint_default.set('limited', 'true')
        
        geom_default = ET.SubElement(default, 'geom')
        geom_default.set('conaffinity', '1')
        geom_default.set('condim', '3')
        geom_default.set('density', '1000.0')
        geom_default.set('friction', '1.5 0.1 0.1')  # Good for dance floor
        geom_default.set('margin', '0.01')
        geom_default.set('rgba', '0.8 0.6 0.4 1')
        
        motor_default = ET.SubElement(default, 'motor')
        motor_default.set('ctrllimited', 'true')
        motor_default.set('ctrlrange', '-200.0 200.0')  # Higher for dynamic moves
        
        # Add assets
        assets = ET.SubElement(root, 'asset')
        
        # Floor texture
        texture1 = ET.SubElement(assets, 'texture')
        texture1.set('name', 'floor_tex')
        texture1.set('type', '2d')
        texture1.set('builtin', 'checker')
        texture1.set('rgb1', '0.1 0.1 0.1')
        texture1.set('rgb2', '0.9 0.9 0.9')
        texture1.set('width', '100')
        texture1.set('height', '100')
        
        # Stage texture
        texture2 = ET.SubElement(assets, 'texture')
        texture2.set('name', 'stage_tex')
        texture2.set('type', '2d')
        texture2.set('builtin', 'flat')
        texture2.set('rgb1', '0.2 0.1 0.3')
        texture2.set('width', '100')
        texture2.set('height', '100')
        
        # Materials
        mat1 = ET.SubElement(assets, 'material')
        mat1.set('name', 'floor_mat')
        mat1.set('texture', 'floor_tex')
        mat1.set('reflectance', '0.5')
        mat1.set('shininess', '0.3')
        
        mat2 = ET.SubElement(assets, 'material')
        mat2.set('name', 'stage_mat')
        mat2.set('texture', 'stage_tex')
        mat2.set('reflectance', '0.8')
        mat2.set('shininess', '0.8')
        
        mat3 = ET.SubElement(assets, 'material')
        mat3.set('name', 'body_mat')
        mat3.set('rgba', '0.8 0.6 0.4 1')
        mat3.set('reflectance', '0.5')
        
        # Add worldbody
        worldbody = ET.SubElement(root, 'worldbody')
        
        # Add lights
        light1 = ET.SubElement(worldbody, 'light')
        light1.set('name', 'spotlight1')
        light1.set('pos', '0 0 10')
        light1.set('dir', '0 0 -1')
        light1.set('diffuse', '1 1 1')
        light1.set('specular', '0.5 0.5 0.5')
        light1.set('cutoff', '90')
        light1.set('exponent', '10')
        
        light2 = ET.SubElement(worldbody, 'light')
        light2.set('name', 'ambient')
        light2.set('pos', '0 0 5')
        light2.set('dir', '0 0 -1')
        light2.set('diffuse', '0.5 0.5 0.6')
        light2.set('specular', '0 0 0')
        
        # Dance floor (circular stage)
        floor = ET.SubElement(worldbody, 'geom')
        floor.set('name', 'dance_floor')
        floor.set('type', 'cylinder')
        floor.set('pos', '0 0 0')
        floor.set('size', f'{self.floor_radius} 0.1')
        floor.set('material', 'floor_mat')
        
        # Stage platform
        stage = ET.SubElement(worldbody, 'geom')
        stage.set('name', 'stage')
        stage.set('type', 'cylinder')
        stage.set('pos', f'0 0 {self.stage_height}')
        stage.set('size', f'{self.floor_radius * 0.8} 0.05')
        stage.set('material', 'stage_mat')
        
        # Add disco ball (decorative)
        disco_ball = ET.SubElement(worldbody, 'body')
        disco_ball.set('name', 'disco_ball')
        disco_ball.set('pos', '0 0 8')
        
        disco_geom = ET.SubElement(disco_ball, 'geom')
        disco_geom.set('name', 'disco_ball_geom')
        disco_geom.set('type', 'sphere')
        disco_geom.set('size', '0.5')
        disco_geom.set('rgba', '0.9 0.9 0.9 1')
        disco_geom.set('conaffinity', '0')
        disco_geom.set('contype', '0')
        
        # Add humanoid dancer
        humanoid = ET.SubElement(worldbody, 'body')
        humanoid.set('name', 'torso')
        humanoid.set('pos', '0 0 1.8')
        
        # Torso
        torso_geom = ET.SubElement(humanoid, 'geom')
        torso_geom.set('name', 'torso_geom')
        torso_geom.set('type', 'capsule')
        torso_geom.set('size', '0.15 0.3')
        torso_geom.set('material', 'body_mat')
        
        # Head
        head_body = ET.SubElement(humanoid, 'body')
        head_body.set('name', 'head')
        head_body.set('pos', '0 0 0.5')
        
        head_joint = ET.SubElement(head_body, 'joint')
        head_joint.set('name', 'neck_x')
        head_joint.set('type', 'hinge')
        head_joint.set('axis', '1 0 0')
        head_joint.set('range', '-30 30')
        
        head_joint2 = ET.SubElement(head_body, 'joint')
        head_joint2.set('name', 'neck_y')
        head_joint2.set('type', 'hinge')
        head_joint2.set('axis', '0 1 0')
        head_joint2.set('range', '-30 30')
        
        head_geom = ET.SubElement(head_body, 'geom')
        head_geom.set('name', 'head_geom')
        head_geom.set('type', 'sphere')
        head_geom.set('size', '0.12')
        head_geom.set('material', 'body_mat')
        
        # Abdomen joints
        abdomen_x = ET.SubElement(humanoid, 'joint')
        abdomen_x.set('name', 'abdomen_x')
        abdomen_x.set('type', 'hinge')
        abdomen_x.set('axis', '1 0 0')
        abdomen_x.set('range', '-45 45')
        
        abdomen_y = ET.SubElement(humanoid, 'joint')
        abdomen_y.set('name', 'abdomen_y')
        abdomen_y.set('type', 'hinge')
        abdomen_y.set('axis', '0 1 0')
        abdomen_y.set('range', '-75 30')
        
        abdomen_z = ET.SubElement(humanoid, 'joint')
        abdomen_z.set('name', 'abdomen_z')
        abdomen_z.set('type', 'hinge')
        abdomen_z.set('axis', '0 0 1')
        abdomen_z.set('range', '-60 60')
        
        # Right arm
        right_shoulder = ET.SubElement(humanoid, 'body')
        right_shoulder.set('name', 'right_shoulder')
        right_shoulder.set('pos', '0.3 0 0.3')
        
        rshoulder1 = ET.SubElement(right_shoulder, 'joint')
        rshoulder1.set('name', 'right_shoulder1')
        rshoulder1.set('type', 'hinge')
        rshoulder1.set('axis', '1 0 0')
        rshoulder1.set('range', '-85 60')
        
        rshoulder2 = ET.SubElement(right_shoulder, 'joint')
        rshoulder2.set('name', 'right_shoulder2')
        rshoulder2.set('type', 'hinge')
        rshoulder2.set('axis', '0 1 0')
        rshoulder2.set('range', '-85 85')
        
        rupperarm_geom = ET.SubElement(right_shoulder, 'geom')
        rupperarm_geom.set('name', 'right_upper_arm')
        rupperarm_geom.set('type', 'capsule')
        rupperarm_geom.set('fromto', '0 0 0 0 0 -0.3')
        rupperarm_geom.set('size', '0.05')
        rupperarm_geom.set('material', 'body_mat')
        
        # Right elbow
        right_elbow_body = ET.SubElement(right_shoulder, 'body')
        right_elbow_body.set('name', 'right_elbow_body')
        right_elbow_body.set('pos', '0 0 -0.3')
        
        relbow = ET.SubElement(right_elbow_body, 'joint')
        relbow.set('name', 'right_elbow')
        relbow.set('type', 'hinge')
        relbow.set('axis', '0 1 0')
        relbow.set('range', '-90 90')
        
        rlowerarm_geom = ET.SubElement(right_elbow_body, 'geom')
        rlowerarm_geom.set('name', 'right_lower_arm')
        rlowerarm_geom.set('type', 'capsule')
        rlowerarm_geom.set('fromto', '0 0 0 0 0 -0.25')
        rlowerarm_geom.set('size', '0.04')
        rlowerarm_geom.set('material', 'body_mat')
        
        # Right wrist
        right_wrist_body = ET.SubElement(right_elbow_body, 'body')
        right_wrist_body.set('name', 'right_wrist_body')
        right_wrist_body.set('pos', '0 0 -0.25')
        
        rwrist_x = ET.SubElement(right_wrist_body, 'joint')
        rwrist_x.set('name', 'right_wrist_x')
        rwrist_x.set('type', 'hinge')
        rwrist_x.set('axis', '1 0 0')
        rwrist_x.set('range', '-30 30')
        
        rwrist_y = ET.SubElement(right_wrist_body, 'joint')
        rwrist_y.set('name', 'right_wrist_y')
        rwrist_y.set('type', 'hinge')
        rwrist_y.set('axis', '0 1 0')
        rwrist_y.set('range', '-30 30')
        
        rwrist_z = ET.SubElement(right_wrist_body, 'joint')
        rwrist_z.set('name', 'right_wrist_z')
        rwrist_z.set('type', 'hinge')
        rwrist_z.set('axis', '0 0 1')
        rwrist_z.set('range', '-30 30')
        
        rhand_geom = ET.SubElement(right_wrist_body, 'geom')
        rhand_geom.set('name', 'right_hand')
        rhand_geom.set('type', 'box')
        rhand_geom.set('size', '0.04 0.02 0.08')
        rhand_geom.set('material', 'body_mat')
        
        # Left arm (mirror of right arm)
        left_shoulder = ET.SubElement(humanoid, 'body')
        left_shoulder.set('name', 'left_shoulder')
        left_shoulder.set('pos', '-0.3 0 0.3')
        
        lshoulder1 = ET.SubElement(left_shoulder, 'joint')
        lshoulder1.set('name', 'left_shoulder1')
        lshoulder1.set('type', 'hinge')
        lshoulder1.set('axis', '1 0 0')
        lshoulder1.set('range', '-60 85')
        
        lshoulder2 = ET.SubElement(left_shoulder, 'joint')
        lshoulder2.set('name', 'left_shoulder2')
        lshoulder2.set('type', 'hinge')
        lshoulder2.set('axis', '0 1 0')
        lshoulder2.set('range', '-85 85')
        
        lupperarm_geom = ET.SubElement(left_shoulder, 'geom')
        lupperarm_geom.set('name', 'left_upper_arm')
        lupperarm_geom.set('type', 'capsule')
        lupperarm_geom.set('fromto', '0 0 0 0 0 -0.3')
        lupperarm_geom.set('size', '0.05')
        lupperarm_geom.set('material', 'body_mat')
        
        # Left elbow
        left_elbow_body = ET.SubElement(left_shoulder, 'body')
        left_elbow_body.set('name', 'left_elbow_body')
        left_elbow_body.set('pos', '0 0 -0.3')
        
        lelbow = ET.SubElement(left_elbow_body, 'joint')
        lelbow.set('name', 'left_elbow')
        lelbow.set('type', 'hinge')
        lelbow.set('axis', '0 1 0')
        lelbow.set('range', '-90 90')
        
        llowerarm_geom = ET.SubElement(left_elbow_body, 'geom')
        llowerarm_geom.set('name', 'left_lower_arm')
        llowerarm_geom.set('type', 'capsule')
        llowerarm_geom.set('fromto', '0 0 0 0 0 -0.25')
        llowerarm_geom.set('size', '0.04')
        llowerarm_geom.set('material', 'body_mat')
        
        # Left wrist
        left_wrist_body = ET.SubElement(left_elbow_body, 'body')
        left_wrist_body.set('name', 'left_wrist_body')
        left_wrist_body.set('pos', '0 0 -0.25')
        
        lwrist_x = ET.SubElement(left_wrist_body, 'joint')
        lwrist_x.set('name', 'left_wrist_x')
        lwrist_x.set('type', 'hinge')
        lwrist_x.set('axis', '1 0 0')
        lwrist_x.set('range', '-30 30')
        
        lwrist_y = ET.SubElement(left_wrist_body, 'joint')
        lwrist_y.set('name', 'left_wrist_y')
        lwrist_y.set('type', 'hinge')
        lwrist_y.set('axis', '0 1 0')
        lwrist_y.set('range', '-30 30')
        
        lwrist_z = ET.SubElement(left_wrist_body, 'joint')
        lwrist_z.set('name', 'left_wrist_z')
        lwrist_z.set('type', 'hinge')
        lwrist_z.set('axis', '0 0 1')
        lwrist_z.set('range', '-30 30')
        
        lhand_geom = ET.SubElement(left_wrist_body, 'geom')
        lhand_geom.set('name', 'left_hand')
        lhand_geom.set('type', 'box')
        lhand_geom.set('size', '0.04 0.02 0.08')
        lhand_geom.set('material', 'body_mat')
        
        # Right leg
        right_hip = ET.SubElement(humanoid, 'body')
        right_hip.set('name', 'right_hip')
        right_hip.set('pos', '0.15 0 -0.3')
        
        rhip_x = ET.SubElement(right_hip, 'joint')
        rhip_x.set('name', 'right_hip_x')
        rhip_x.set('type', 'hinge')
        rhip_x.set('axis', '1 0 0')
        rhip_x.set('range', '-30 120')
        
        rhip_y = ET.SubElement(right_hip, 'joint')
        rhip_y.set('name', 'right_hip_y')
        rhip_y.set('type', 'hinge')
        rhip_y.set('axis', '0 1 0')
        rhip_y.set('range', '-120 30')
        
        rhip_z = ET.SubElement(right_hip, 'joint')
        rhip_z.set('name', 'right_hip_z')
        rhip_z.set('type', 'hinge')
        rhip_z.set('axis', '0 0 1')
        rhip_z.set('range', '-30 30')
        
        rthigh_geom = ET.SubElement(right_hip, 'geom')
        rthigh_geom.set('name', 'right_thigh')
        rthigh_geom.set('type', 'capsule')
        rthigh_geom.set('fromto', '0 0 0 0 0 -0.4')
        rthigh_geom.set('size', '0.06')
        rthigh_geom.set('material', 'body_mat')
        
        # Right knee
        right_knee_body = ET.SubElement(right_hip, 'body')
        right_knee_body.set('name', 'right_knee_body')
        right_knee_body.set('pos', '0 0 -0.4')
        
        rknee = ET.SubElement(right_knee_body, 'joint')
        rknee.set('name', 'right_knee')
        rknee.set('type', 'hinge')
        rknee.set('axis', '0 1 0')
        rknee.set('range', '-150 0')
        
        rshin_geom = ET.SubElement(right_knee_body, 'geom')
        rshin_geom.set('name', 'right_shin')
        rshin_geom.set('type', 'capsule')
        rshin_geom.set('fromto', '0 0 0 0 0 -0.35')
        rshin_geom.set('size', '0.05')
        rshin_geom.set('material', 'body_mat')
        
        # Right ankle
        right_ankle_body = ET.SubElement(right_knee_body, 'body')
        right_ankle_body.set('name', 'right_ankle_body')
        right_ankle_body.set('pos', '0 0 -0.35')
        
        rankle_x = ET.SubElement(right_ankle_body, 'joint')
        rankle_x.set('name', 'right_ankle_x')
        rankle_x.set('type', 'hinge')
        rankle_x.set('axis', '1 0 0')
        rankle_x.set('range', '-30 30')
        
        rankle_y = ET.SubElement(right_ankle_body, 'joint')
        rankle_y.set('name', 'right_ankle_y')
        rankle_y.set('type', 'hinge')
        rankle_y.set('axis', '0 1 0')
        rankle_y.set('range', '-50 50')
        
        rfoot_geom = ET.SubElement(right_ankle_body, 'geom')
        rfoot_geom.set('name', 'right_foot')
        rfoot_geom.set('type', 'box')
        rfoot_geom.set('pos', '0 0.05 -0.02')
        rfoot_geom.set('size', '0.06 0.12 0.02')
        rfoot_geom.set('material', 'body_mat')
        
        # Left leg (mirror of right leg)
        left_hip = ET.SubElement(humanoid, 'body')
        left_hip.set('name', 'left_hip')
        left_hip.set('pos', '-0.15 0 -0.3')
        
        lhip_x = ET.SubElement(left_hip, 'joint')
        lhip_x.set('name', 'left_hip_x')
        lhip_x.set('type', 'hinge')
        lhip_x.set('axis', '1 0 0')
        lhip_x.set('range', '-30 120')
        
        lhip_y = ET.SubElement(left_hip, 'joint')
        lhip_y.set('name', 'left_hip_y')
        lhip_y.set('type', 'hinge')
        lhip_y.set('axis', '0 1 0')
        lhip_y.set('range', '-120 30')
        
        lhip_z = ET.SubElement(left_hip, 'joint')
        lhip_z.set('name', 'left_hip_z')
        lhip_z.set('type', 'hinge')
        lhip_z.set('axis', '0 0 1')
        lhip_z.set('range', '-30 30')
        
        lthigh_geom = ET.SubElement(left_hip, 'geom')
        lthigh_geom.set('name', 'left_thigh')
        lthigh_geom.set('type', 'capsule')
        lthigh_geom.set('fromto', '0 0 0 0 0 -0.4')
        lthigh_geom.set('size', '0.06')
        lthigh_geom.set('material', 'body_mat')
        
        # Left knee
        left_knee_body = ET.SubElement(left_hip, 'body')
        left_knee_body.set('name', 'left_knee_body')
        left_knee_body.set('pos', '0 0 -0.4')
        
        lknee = ET.SubElement(left_knee_body, 'joint')
        lknee.set('name', 'left_knee')
        lknee.set('type', 'hinge')
        lknee.set('axis', '0 1 0')
        lknee.set('range', '-150 0')
        
        lshin_geom = ET.SubElement(left_knee_body, 'geom')
        lshin_geom.set('name', 'left_shin')
        lshin_geom.set('type', 'capsule')
        lshin_geom.set('fromto', '0 0 0 0 0 -0.35')
        lshin_geom.set('size', '0.05')
        lshin_geom.set('material', 'body_mat')
        
        # Left ankle
        left_ankle_body = ET.SubElement(left_knee_body, 'body')
        left_ankle_body.set('name', 'left_ankle_body')
        left_ankle_body.set('pos', '0 0 -0.35')
        
        lankle_x = ET.SubElement(left_ankle_body, 'joint')
        lankle_x.set('name', 'left_ankle_x')
        lankle_x.set('type', 'hinge')
        lankle_x.set('axis', '1 0 0')
        lankle_x.set('range', '-30 30')
        
        lankle_y = ET.SubElement(left_ankle_body, 'joint')
        lankle_y.set('name', 'left_ankle_y')
        lankle_y.set('type', 'hinge')
        lankle_y.set('axis', '0 1 0')
        lankle_y.set('range', '-50 50')
        
        lfoot_geom = ET.SubElement(left_ankle_body, 'geom')
        lfoot_geom.set('name', 'left_foot')
        lfoot_geom.set('type', 'box')
        lfoot_geom.set('pos', '0 0.05 -0.02')
        lfoot_geom.set('size', '0.06 0.12 0.02')
        lfoot_geom.set('material', 'body_mat')
        
        # Add actuators for all joints
        actuators = ET.SubElement(root, 'actuator')
        
        joint_names = [
            'abdomen_x', 'abdomen_y', 'abdomen_z',
            'neck_x', 'neck_y',
            'right_shoulder1', 'right_shoulder2', 'right_elbow',
            'right_wrist_x', 'right_wrist_y', 'right_wrist_z',
            'left_shoulder1', 'left_shoulder2', 'left_elbow',
            'left_wrist_x', 'left_wrist_y', 'left_wrist_z',
            'right_hip_x', 'right_hip_y', 'right_hip_z',
            'right_knee', 'right_ankle_x', 'right_ankle_y',
            'left_hip_x', 'left_hip_y', 'left_hip_z',
            'left_knee', 'left_ankle_x', 'left_ankle_y'
        ]
        
        for joint_name in joint_names:
            motor = ET.SubElement(actuators, 'motor')
            motor.set('name', f'{joint_name}_motor')
            motor.set('joint', joint_name)
            motor.set('gear', '100')
            motor.set('ctrllimited', 'true')
            motor.set('ctrlrange', '-200.0 200.0')
        
        # Add sensors
        sensors = ET.SubElement(root, 'sensor')
        
        # Torso sensors
        torso_acc = ET.SubElement(sensors, 'accelerometer')
        torso_acc.set('name', 'torso_accel')
        torso_acc.set('site', 'torso_site')
        
        torso_gyro = ET.SubElement(sensors, 'gyro')
        torso_gyro.set('name', 'torso_gyro')
        torso_gyro.set('site', 'torso_site')
        
        # Add sites for sensors
        torso_site = ET.SubElement(humanoid, 'site')
        torso_site.set('name', 'torso_site')
        torso_site.set('pos', '0 0 0')
        torso_site.set('size', '0.01')
        
        # Convert to string
        self.xml_string = ET.tostring(root, encoding='unicode')
    
    def _get_model_indices(self):
        """Get indices for important model elements."""
        
        # Joint names for tracking
        self.joint_names = [
            'abdomen_x', 'abdomen_y', 'abdomen_z',
            'neck_x', 'neck_y',
            'right_shoulder1', 'right_shoulder2', 'right_elbow',
            'right_wrist_x', 'right_wrist_y', 'right_wrist_z',
            'left_shoulder1', 'left_shoulder2', 'left_elbow',
            'left_wrist_x', 'left_wrist_y', 'left_wrist_z',
            'right_hip_x', 'right_hip_y', 'right_hip_z',
            'right_knee', 'right_ankle_x', 'right_ankle_y',
            'left_hip_x', 'left_hip_y', 'left_hip_z',
            'left_knee', 'left_ankle_x', 'left_ankle_y'
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
            self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'head')
            self.disco_ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'disco_ball')
        except:
            print("Warning: Some bodies not found")
        
        # Geom indices for contact detection
        try:
            self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'right_foot')
            self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'left_foot')
            self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'dance_floor')
            self.stage_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'stage')
        except:
            print("Warning: Some geoms not found")
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        
        # Action space: torque control for all joints
        self.action_space = spaces.Box(
            low=-200.0,
            high=200.0,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # Observation space components
        obs_dim = (
            self.num_joints * 2 +  # Joint positions and velocities
            4 +  # Torso orientation quaternion
            6 +  # Torso linear and angular velocity
            3 +  # Center of mass position
            2 +  # Foot contact indicators
            3 +  # Target pose difference
            1 +  # Beat phase (0-1)
            1 +  # Time until next beat
            len(self.dance_moves) +  # One-hot encoding of current move
            1 +  # Combo multiplier
            1 +  # Crowd excitement
            3 +  # Spotlight position relative to dancer
            1    # Energy remaining
        )
        
        # Create observation space
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
        self.beat_count = 0
        self.current_measure = 0
        self.time_since_last_beat = 0.0
        
        # Reset performance metrics
        self.performance_score = 0.0
        self.combo_multiplier = 1.0
        self.perfect_moves = 0
        self.good_moves = 0
        self.missed_beats = 0
        
        # Reset crowd excitement
        self.crowd_excitement = 0.5
        self.applause_level = 0.0
        
        # Generate new dance sequence
        self._generate_dance_sequence()
        self.current_move_idx = 0
        self.move_start_time = 0.0
        
        # Reset episode stats
        self.episode_stats = {
            'total_score': 0,
            'perfect_moves': 0,
            'good_moves': 0,
            'missed_beats': 0,
            'longest_combo': 0,
            'energy_used': 0.0,
            'time_on_beat': 0.0,
            'creativity_score': 0.0,
            'crowd_rating': 0.0
        }
        
        # Set initial pose
        self._set_initial_pose()
        
        # Step simulation to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Get initial observation
        obs = self._get_observation()
        
        # Store previous state
        self.prev_joint_pos = self.data.qpos[7:].copy()  # Skip root position/orientation
        self.prev_joint_vel = self.data.qvel[6:].copy()  # Skip root velocities
        self.prev_robot_pos = self._get_robot_position().copy()
        self.prev_robot_quat = self._get_robot_orientation().copy()
        self.move_history = []
        
        info = {
            'episode_stats': self.episode_stats.copy(),
            'current_move': self.dance_moves[list(self.dance_moves.keys())[0]],
            'beat_phase': 0.0,
            'combo_multiplier': self.combo_multiplier
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply actions to actuators
        self.data.ctrl[:] = action
        
        # Update rhythm tracking
        self._update_rhythm()
        
        # Update visual effects
        self._update_visual_effects()
        
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
        
        # Update crowd excitement based on performance
        self._update_crowd_excitement()
        
        # Check for move transitions
        self._check_move_transition()
        
        # Prepare info
        current_move_name = list(self.dance_moves.keys())[self.current_move_idx % len(self.dance_moves)]
        info = {
            'episode_stats': self.episode_stats.copy(),
            'current_move': self.dance_moves[current_move_name],
            'beat_phase': self.time_since_last_beat / self.beat_interval,
            'combo_multiplier': self.combo_multiplier,
            'crowd_excitement': self.crowd_excitement,
            'performance_score': self.performance_score
        }
        
        # Update previous state
        self.prev_joint_pos = self.data.qpos[7:].copy()
        self.prev_joint_vel = self.data.qvel[6:].copy()
        self.prev_robot_pos = self._get_robot_position().copy()
        self.prev_robot_quat = self._get_robot_orientation().copy()
        
        # Render if needed (following MuJoCo visualization fix)
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _generate_dance_sequence(self):
        """Generate a random dance sequence for the episode."""
        sequence_length = 20  # Number of moves in sequence
        self.dance_sequence = []
        
        move_names = list(self.dance_moves.keys())
        for _ in range(sequence_length):
            move = self.np_random.choice(move_names)
            duration = self.np_random.uniform(1.0, 3.0)  # Variable move durations
            self.dance_sequence.append({'move': move, 'duration': duration})
    
    def _set_initial_pose(self):
        """Set the humanoid in an initial standing pose."""
        
        # Set initial position (standing on stage)
        self.data.qpos[0] = 0.0  # x
        self.data.qpos[1] = 0.0  # y
        self.data.qpos[2] = self.robot_height  # z (standing height)
        
        # Set upright orientation (quaternion)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        # Set joints to neutral positions
        for i, joint_idx in enumerate(self.joint_indices):
            qpos_idx = 7 + i
            if qpos_idx < len(self.data.qpos) and joint_idx < len(self.model.jnt_range):
                self.data.qpos[qpos_idx] = 0.0  # Neutral position
    
    def _update_rhythm(self):
        """Update rhythm tracking and beat detection."""
        
        # Update time since last beat
        self.time_since_last_beat += self.dt
        
        # Check for beat
        if self.time_since_last_beat >= self.beat_interval:
            self.time_since_last_beat -= self.beat_interval
            self.beat_count += 1
            
            # Update measure (4 beats per measure)
            if self.beat_count % 4 == 0:
                self.current_measure += 1
    
    def _update_visual_effects(self):
        """Update visual effects like disco ball rotation and spotlight."""
        
        # Rotate disco ball
        self.disco_ball_rotation += 0.5 * self.dt
        if self.disco_ball_rotation > 2 * np.pi:
            self.disco_ball_rotation -= 2 * np.pi
        
        # Move spotlight to follow dancer with some lag
        robot_pos = self._get_robot_position()
        target_spotlight = np.array([robot_pos[0], robot_pos[1], 5.0])
        self.spotlight_position += 0.1 * (target_spotlight - self.spotlight_position)
        
        # Update spotlight color based on beat
        beat_phase = self.time_since_last_beat / self.beat_interval
        intensity = 0.7 + 0.3 * np.sin(2 * np.pi * beat_phase)
        self.spotlight_color = np.array([intensity, intensity * 0.8, intensity * 0.9])
    
    def _check_move_transition(self):
        """Check if it's time to transition to the next move."""
        
        current_time = self.current_step * self.dt
        move_elapsed = current_time - self.move_start_time
        
        if self.current_move_idx < len(self.dance_sequence):
            current_move_duration = self.dance_sequence[self.current_move_idx]['duration']
            
            if move_elapsed >= current_move_duration:
                # Transition to next move
                self.current_move_idx += 1
                self.move_start_time = current_time
                
                # Add to move history
                if self.current_move_idx < len(self.dance_sequence):
                    self.move_history.append(self.dance_sequence[self.current_move_idx]['move'])
    
    def _update_crowd_excitement(self):
        """Update crowd excitement based on performance."""
        
        # Factors affecting crowd excitement
        on_beat_factor = 0.0
        if self.time_since_last_beat < 0.1 or self.time_since_last_beat > self.beat_interval - 0.1:
            on_beat_factor = 0.1
        
        # Update based on combo
        combo_factor = min(self.combo_multiplier / 10.0, 1.0) * 0.2
        
        # Update based on move difficulty
        if self.current_move_idx < len(self.dance_sequence):
            move_name = self.dance_sequence[self.current_move_idx]['move']
            difficulty = self.dance_moves[move_name]['difficulty']
            difficulty_factor = difficulty / 4.0 * 0.1
        else:
            difficulty_factor = 0.0
        
        # Apply excitement change
        excitement_change = (on_beat_factor + combo_factor + difficulty_factor) * 0.01
        self.crowd_excitement = np.clip(self.crowd_excitement + excitement_change, 0.0, 1.0)
        
        # Decay excitement slightly
        self.crowd_excitement *= 0.999
        
        # Update applause level
        self.applause_level = self.crowd_excitement * 100.0
    
    def _update_episode_stats(self):
        """Update episode statistics."""
        
        # Update energy used
        energy = np.sum(np.abs(self.data.ctrl))
        self.episode_stats['energy_used'] += energy * self.dt
        
        # Update time on beat
        beat_phase = self.time_since_last_beat / self.beat_interval
        if beat_phase < 0.1 or beat_phase > 0.9:
            self.episode_stats['time_on_beat'] += self.dt
        
        # Update longest combo
        self.episode_stats['longest_combo'] = max(
            self.episode_stats['longest_combo'],
            int(self.combo_multiplier)
        )
        
        # Update crowd rating
        self.episode_stats['crowd_rating'] = self.crowd_excitement
        
        # Update total score
        self.episode_stats['total_score'] = self.performance_score
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        
        obs = []
        
        # Joint positions (normalized)
        for i in range(self.num_joints):
            qpos_idx = 7 + i  # Skip root DOFs
            if i < len(self.joint_indices) and qpos_idx < len(self.data.qpos):
                joint_idx = self.joint_indices[i]
                pos = self.data.qpos[qpos_idx]
                if joint_idx < len(self.model.jnt_range):
                    joint_range = self.model.jnt_range[joint_idx]
                    if joint_range[0] < joint_range[1]:
                        normalized_pos = 2 * (pos - joint_range[0]) / (joint_range[1] - joint_range[0]) - 1
                        obs.append(np.clip(normalized_pos, -1.0, 1.0))
                    else:
                        obs.append(0.0)
                else:
                    obs.append(0.0)
            else:
                obs.append(0.0)
        
        # Joint velocities (normalized)
        for i in range(self.num_joints):
            if i < len(self.data.qvel) - 6:
                vel = self.data.qvel[6 + i]  # Skip root DOFs
                normalized_vel = np.clip(vel / 10.0, -1.0, 1.0)
                obs.append(normalized_vel)
            else:
                obs.append(0.0)
        
        # Torso orientation quaternion
        torso_quat = self._get_robot_orientation()
        obs.extend(torso_quat)
        
        # Torso linear velocity (normalized)
        torso_vel = self.data.qvel[:3]
        normalized_torso_vel = np.clip(torso_vel / 5.0, -1.0, 1.0)
        obs.extend(normalized_torso_vel)
        
        # Torso angular velocity (normalized)
        torso_angvel = self.data.qvel[3:6]
        normalized_torso_angvel = np.clip(torso_angvel / 10.0, -1.0, 1.0)
        obs.extend(normalized_torso_angvel)
        
        # Center of mass position (normalized)
        com_pos = self._get_robot_com_position()
        normalized_com_pos = np.clip(com_pos / 10.0, -1.0, 1.0)
        obs.extend(normalized_com_pos)
        
        # Foot contact indicators
        foot_contacts = self._get_foot_contacts()
        obs.extend(foot_contacts)
        
        # Target pose difference (placeholder - would be actual target in full implementation)
        target_diff = np.zeros(3)
        obs.extend(target_diff)
        
        # Beat phase (0-1)
        beat_phase = self.time_since_last_beat / self.beat_interval
        obs.append(beat_phase)
        
        # Time until next beat (normalized)
        time_to_beat = (self.beat_interval - self.time_since_last_beat) / self.beat_interval
        obs.append(time_to_beat)
        
        # One-hot encoding of current move
        move_encoding = np.zeros(len(self.dance_moves))
        if self.current_move_idx < len(self.dance_sequence):
            move_name = self.dance_sequence[self.current_move_idx]['move']
            move_idx = list(self.dance_moves.keys()).index(move_name)
            move_encoding[move_idx] = 1.0
        obs.extend(move_encoding)
        
        # Combo multiplier (normalized)
        normalized_combo = np.clip(self.combo_multiplier / 10.0, 0.0, 1.0)
        obs.append(normalized_combo)
        
        # Crowd excitement
        obs.append(self.crowd_excitement)
        
        # Spotlight position relative to dancer
        robot_pos = self._get_robot_position()
        rel_spotlight = self.spotlight_position - robot_pos
        normalized_spotlight = np.clip(rel_spotlight / 10.0, -1.0, 1.0)
        obs.extend(normalized_spotlight)
        
        # Energy remaining (simplified)
        energy_remaining = 1.0 - min(self.episode_stats['energy_used'] / 1000.0, 1.0)
        obs.append(energy_remaining)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current step."""
        
        reward = 0.0
        
        # Rhythm reward - bonus for moving on the beat
        beat_phase = self.time_since_last_beat / self.beat_interval
        if beat_phase < 0.1 or beat_phase > 0.9:  # Near the beat
            # Check if robot is moving significantly
            joint_vel = self.data.qvel[6:]  # Skip root velocities
            movement_magnitude = np.linalg.norm(joint_vel)
            if movement_magnitude > 1.0:
                reward += self.rhythm_reward
                self.combo_multiplier = min(self.combo_multiplier + 0.1, 10.0)
            else:
                self.combo_multiplier = max(self.combo_multiplier - 0.05, 1.0)
        
        # Balance reward
        if self._is_robot_upright():
            reward += self.balance_reward
            
            # Additional reward for maintaining balance while moving
            if self.prev_joint_vel is not None:
                vel_change = np.linalg.norm(self.data.qvel[6:] - self.prev_joint_vel)
                if vel_change > 0.5:  # Moving
                    reward += self.balance_reward * 0.5
        
        # Smoothness reward - penalize jerky movements
        if self.prev_joint_vel is not None:
            jerk = np.linalg.norm(self.data.qvel[6:] - self.prev_joint_vel)
            smoothness = np.exp(-0.1 * jerk)
            reward += self.smoothness_reward * smoothness
        
        # Style bonus - reward for variety in movement
        if len(self.move_history) > 2:
            recent_moves = self.move_history[-3:]
            if len(set(recent_moves)) == 3:  # All different
                reward += self.style_bonus
        
        # Move completion reward
        current_time = self.current_step * self.dt
        move_elapsed = current_time - self.move_start_time
        if self.current_move_idx < len(self.dance_sequence):
            move_duration = self.dance_sequence[self.current_move_idx]['duration']
            if move_elapsed > move_duration * 0.8:  # Near completion
                move_name = self.dance_sequence[self.current_move_idx]['move']
                difficulty = self.dance_moves[move_name]['difficulty']
                reward += self.move_completion_reward * difficulty
        
        # Creativity bonus - reward for using full range of motion
        joint_range_used = 0
        for i, joint_idx in enumerate(self.joint_indices):
            qpos_idx = 7 + i
            if qpos_idx < len(self.data.qpos) and joint_idx < len(self.model.jnt_range):
                pos = self.data.qpos[qpos_idx]
                joint_range = self.model.jnt_range[joint_idx]
                if joint_range[0] < joint_range[1]:
                    range_pct = abs(pos - (joint_range[0] + joint_range[1]) / 2) / (joint_range[1] - joint_range[0])
                    joint_range_used += range_pct
        
        if joint_range_used > 5.0:  # Using many joints actively
            reward += self.creativity_bonus * 0.1
        
        # Energy penalty
        energy_cost = np.sum(np.square(action))
        reward += self.energy_penalty * energy_cost
        
        # Fall penalty
        if not self._is_robot_upright():
            reward += self.fall_penalty
            self.combo_multiplier = 1.0  # Reset combo
        
        # Off-beat penalty
        if beat_phase > 0.2 and beat_phase < 0.8:  # Mid-beat
            movement_magnitude = np.linalg.norm(self.data.qvel[6:])
            if movement_magnitude > 3.0:  # Moving too much off-beat
                reward += self.off_beat_penalty * 0.1
        
        # Apply combo multiplier to positive rewards
        if reward > 0:
            reward *= self.combo_multiplier
        
        # Update performance score
        self.performance_score += reward
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        
        # Robot fell and can't recover
        if not self._is_robot_upright():
            # Check if robot has been down for more than 2 seconds
            if not hasattr(self, 'fall_start_step'):
                self.fall_start_step = self.current_step
            elif self.current_step - self.fall_start_step > 120:  # 2 seconds at 60 Hz
                return True
        else:
            # Reset fall counter if upright
            if hasattr(self, 'fall_start_step'):
                delattr(self, 'fall_start_step')
        
        # Robot went out of bounds
        robot_pos = self._get_robot_position()
        distance_from_center = np.linalg.norm(robot_pos[:2])
        if distance_from_center > self.floor_radius * 1.5:
            return True
        
        # Robot is too high or too low
        if robot_pos[2] < 0.0 or robot_pos[2] > 5.0:
            return True
        
        return False
    
    # Helper methods
    def _get_robot_position(self) -> np.ndarray:
        """Get robot torso position."""
        return self.data.xpos[self.torso_id].copy()
    
    def _get_robot_orientation(self) -> np.ndarray:
        """Get robot torso orientation quaternion."""
        return self.data.xquat[self.torso_id].copy()
    
    def _get_robot_com_position(self) -> np.ndarray:
        """Get robot center of mass position."""
        return self.data.subtree_com[self.torso_id].copy()
    
    def _get_foot_contacts(self) -> np.ndarray:
        """Get foot contact indicators."""
        contacts = np.zeros(2)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Right foot contact
            if ((contact.geom1 == self.right_foot_id and contact.geom2 in [self.floor_id, self.stage_id]) or
                (contact.geom2 == self.right_foot_id and contact.geom1 in [self.floor_id, self.stage_id])):
                contacts[0] = 1.0
            
            # Left foot contact
            if ((contact.geom1 == self.left_foot_id and contact.geom2 in [self.floor_id, self.stage_id]) or
                (contact.geom2 == self.left_foot_id and contact.geom1 in [self.floor_id, self.stage_id])):
                contacts[1] = 1.0
        
        return contacts
    
    def _is_robot_upright(self) -> bool:
        """Check if robot is upright."""
        
        quat = self._get_robot_orientation()
        
        # Convert quaternion to rotation matrix
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, quat.astype(np.float64))
        rot_mat = rot_mat.reshape(3, 3)
        
        # Check if z-axis is pointing up
        up_vector = rot_mat[:, 2]
        upright_threshold = 0.7  # cos(45 degrees)
        
        return up_vector[2] > upright_threshold
    
    def _init_viewer(self):
        """Initialize MuJoCo viewer (following visualization fix)."""
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("🎮 MuJoCo Dancing Environment viewer opened successfully!")
        except Exception as e:
            print(f"⚠️ Could not open viewer: {e}")
            self.viewer = None
    
    def render(self):
        """Render the environment (following MuJoCo visualization fix)."""
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
            id='HumanoidDancing-v0',
            entry_point='humanoid_dancing_env.dancing_env:HumanoidDancingEnv',
            max_episode_steps=3600,
            reward_threshold=5000.0,
        )
    except gym.error.Error:
        # Environment already registered
        pass


if __name__ == "__main__":
    # Test the environment
    register_env()
    env = HumanoidDancingEnv(render_mode='human')
    
    obs, info = env.reset()
    print(f"🎵 Humanoid Dancing Environment initialized!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Current BPM: {env.bpm}")
    
    for i in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 60 == 0:  # Print every second
            print(f"Step {i}: Reward = {reward:.2f}, Combo = {info['combo_multiplier']:.1f}x, "
                  f"Crowd = {info['crowd_excitement']:.2%}, Beat Phase = {info['beat_phase']:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Final performance score: {info['performance_score']:.2f}")
            print(f"Episode stats: {info['episode_stats']}")
            obs, info = env.reset()
    
    env.close()
