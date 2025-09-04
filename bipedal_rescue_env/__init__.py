"""
Bipedal Rescue Environment

A MuJoCo-based disaster rescue simulation with a bipedal robot.

Author: Hasnain Fareed
License: MIT (2025)
"""

from .rescue_env import BipedalRescueEnv, register_env

__all__ = ['BipedalRescueEnv', 'register_env']
__version__ = '1.0.0'
