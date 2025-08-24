"""
Quadruped Parkour Environment Package

A MuJoCo-based reinforcement learning environment featuring a quadruped robot
navigating through challenging parkour courses with dynamic obstacles.
"""

from gymnasium.envs.registration import register
from .parkour_env import QuadrupedParkourEnv

__version__ = "1.0.0"
__author__ = "Hasnain Fareed"
__email__ = "hasnain.fareed@example.com"

# Register the environment with Gymnasium
register(
    id='QuadrupedParkour-v0',
    entry_point='quadruped_parkour_env:QuadrupedParkourEnv',
    max_episode_steps=6000,
    reward_threshold=8000.0,
    kwargs={
        'render_mode': None
    }
)

register(
    id='QuadrupedParkour-v1',
    entry_point='quadruped_parkour_env:QuadrupedParkourEnv',
    max_episode_steps=6000,
    reward_threshold=10000.0,
    kwargs={
        'render_mode': 'human'
    }
)

__all__ = ['QuadrupedParkourEnv']
