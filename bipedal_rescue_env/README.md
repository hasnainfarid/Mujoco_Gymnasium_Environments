

<img width="1712" height="900" alt="rescue" src="https://github.com/user-attachments/assets/7d3784b1-927b-4d5e-acc1-42b02563def2" />




# Bipedal Rescue Environment

A sophisticated MuJoCo-based disaster rescue simulation environment for Gymnasium, featuring a bipedal robot navigating through hazardous terrain to rescue victims.

## Features

- **25+ DOF bipedal robot** with highly articulated humanoid design, realistic joint constraints, dual grippers for carrying up to 2 victims, and comprehensive sensor suite (accelerometer, gyroscope, contact sensors)
- **Dynamic disaster environment** with 5 unique victims having different weights and priorities, fire zones, debris fields, unstable structures, and designated safe zone for victim drop-off
- **Advanced rescue mechanics** including victim location, priority-based rescue planning, energy management system with limited battery, and efficient movement planning
- **Realistic physics simulation** using MuJoCo with proper collision detection, dynamic hazard interactions, and authentic robot-environment dynamics
- **Comprehensive visualization** with 3D MuJoCo viewer for real-time physics simulation, 2D Pygame display for mission overview, and real-time mission metrics tracking

## Installation

```bash
cd bipedal_rescue_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from bipedal_rescue_env import BipedalRescueEnv, register_env

# Register the environment
register_env()

# Create the environment
env = gym.make('BipedalRescue-v0', render_mode='human')

# Run a simple episode
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Testing

```bash
python test_rescue.py
```

Run comprehensive tests with both 3D MuJoCo viewer and 2D Pygame visualization.

## Requirements

- gymnasium>=0.28.0
- mujoco>=2.3.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0 (optional)

## Environment Details

- **Action Space**: 25 continuous torque controls for robot joints
- **Observation Space**: 95-dimensional state including joint positions/velocities, robot pose, victim states, and mission status
- **Episode Length**: 2000 steps (20 seconds at 100Hz control frequency)
- **Mission Goals**: Rescue all 5 victims and transport them to safe zones

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025
