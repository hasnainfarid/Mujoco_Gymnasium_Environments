"""
Robotic Arm Assembly Environment Package
MuJoCo-based precision assembly tasks with 7-DOF robotic arm
Author: Hasnain Fareed
Year: 2025
"""

from gymnasium.envs.registration import register
from .assembly_env import RoboticArmAssemblyEnv

# Register the environment
register(
    id='RoboticArmAssembly-v0',
    entry_point='robotic_arm_assembly_env:RoboticArmAssemblyEnv',
    max_episode_steps=150000,
    reward_threshold=8000.0,
)

__version__ = '1.0.0'
__all__ = ['RoboticArmAssemblyEnv']
