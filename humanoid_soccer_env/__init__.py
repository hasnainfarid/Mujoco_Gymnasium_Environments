"""
Humanoid Soccer Environment Package

A comprehensive MuJoCo-based 3D soccer simulation environment for reinforcement learning.
Features a 25-DOF humanoid robot, realistic physics, and challenging soccer gameplay.
"""

from gymnasium.envs.registration import register
from .soccer_env import HumanoidSoccerEnv

# Package information
__version__ = "1.0.0"
__author__ = "Hasnain Fareed"
__email__ = "hasnain.fareed@example.com"
__description__ = "MuJoCo-based humanoid soccer environment for reinforcement learning"

# Register the environment with Gymnasium
register(
    id='HumanoidSoccer-v0',
    entry_point='humanoid_soccer_env.soccer_env:HumanoidSoccerEnv',
    max_episode_steps=2500,
    reward_threshold=8000.0,
    kwargs={
        'render_mode': None
    }
)

# Make the environment class available at package level
__all__ = ['HumanoidSoccerEnv']

def make_env(render_mode=None, **kwargs):
    """
    Convenience function to create a HumanoidSoccerEnv instance.
    
    Args:
        render_mode (str, optional): Rendering mode ('human', 'rgb_array', 'depth_array')
        **kwargs: Additional keyword arguments for the environment
    
    Returns:
        HumanoidSoccerEnv: Configured environment instance
    """
    return HumanoidSoccerEnv(render_mode=render_mode, **kwargs)

def get_env_info():
    """
    Get information about the environment.
    
    Returns:
        dict: Environment information including action/observation space details
    """
    return {
        'name': 'HumanoidSoccer-v0',
        'description': __description__,
        'version': __version__,
        'author': __author__,
        'action_space': 'Box(25,) - Joint torques [-150, 150] Nm',
        'observation_space': 'Box(85,) - Robot state, ball state, environment state',
        'max_episode_steps': 2500,
        'reward_threshold': 8000.0,
        'features': [
            '25-DOF humanoid robot with realistic physics',
            '3D soccer field with goals and boundaries', 
            'Physics-based ball with realistic bounce and friction',
            'Opponent goalkeeper with PID control',
            'Wind effects and environmental variations',
            'Comprehensive reward system for soccer skills',
            'Real-time 3D visualization support'
        ]
    }

# Print package info when imported
print(f"Humanoid Soccer Environment v{__version__} loaded successfully!")
print(f"Use 'import gymnasium as gym; env = gym.make(\"HumanoidSoccer-v0\")' to create environment")
