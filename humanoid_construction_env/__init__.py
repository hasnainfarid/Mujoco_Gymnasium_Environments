"""
Humanoid Construction Environment Package

A comprehensive MuJoCo-based 3D construction simulation environment for reinforcement learning.
Features a humanoid robot operating construction equipment, building structures, and handling materials.
"""

from gymnasium.envs.registration import register
from .construction_env import HumanoidConstructionEnv

# Package information
__version__ = "1.0.0"
__author__ = "Hasnain Fareed"
__email__ = "hasnain.fareed@example.com"
__description__ = "MuJoCo-based humanoid construction environment for reinforcement learning"

# Register the environment with Gymnasium
register(
    id='HumanoidConstruction-v0',
    entry_point='humanoid_construction_env.construction_env:HumanoidConstructionEnv',
    max_episode_steps=3000,
    reward_threshold=10000.0,
    kwargs={
        'render_mode': None
    }
)

# Make the environment class available at package level
__all__ = ['HumanoidConstructionEnv']

def make_env(render_mode=None, **kwargs):
    """
    Convenience function to create a HumanoidConstructionEnv instance.
    
    Args:
        render_mode (str, optional): Rendering mode ('human', 'rgb_array', 'depth_array')
        **kwargs: Additional keyword arguments for the environment
    
    Returns:
        HumanoidConstructionEnv: Configured environment instance
    """
    return HumanoidConstructionEnv(render_mode=render_mode, **kwargs)

def get_env_info():
    """
    Get information about the environment.
    
    Returns:
        dict: Environment information including action/observation space details
    """
    return {
        'name': 'HumanoidConstruction-v0',
        'description': __description__,
        'version': __version__,
        'author': __author__,
        'action_space': 'Box(27,) - Joint torques [-200, 200] Nm',
        'observation_space': 'Box(125,) - Robot state, object states, task progress',
        'max_episode_steps': 3000,
        'reward_threshold': 10000.0,
        'features': [
            '27-DOF humanoid robot with enhanced manipulation capabilities',
            'Construction site with crane, building blocks, and materials',
            'Physics-based construction mechanics',
            'Multiple construction tasks (lifting, placing, building)',
            'Dynamic obstacles and safety hazards',
            'Progressive difficulty levels',
            'Real-time 3D visualization support'
        ]
    }

# Print package info when imported
print(f"Humanoid Construction Environment v{__version__} loaded successfully!")
print(f"Use 'import gymnasium as gym; env = gym.make(\"HumanoidConstruction-v0\")' to create environment")
