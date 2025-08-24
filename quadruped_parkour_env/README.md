# Quadruped Parkour Environment

A comprehensive MuJoCo-based reinforcement learning environment featuring a quadruped robot navigating through challenging parkour courses with dynamic obstacles and terrain variations.

## Features

- **Realistic quadruped locomotion** with Boston Dynamics Spot-style robot featuring 16 degrees of freedom, authentic leg dynamics, spring-damper joints, and contact physics
- **Dynamic parkour course** spanning 100m with stairs, gaps, balance beams, moving platforms, rough terrain, environmental hazards, and procedural generation
- **Comprehensive reward system** with multi-layered structure encouraging course completion, obstacle navigation, energy efficiency, and stable gait patterns
- **Advanced physics simulation** using MuJoCo integration with realistic collision detection, variable terrain friction, and dynamic obstacle interactions
- **Performance analytics** including built-in gait analysis, energy efficiency metrics, and comprehensive testing suite with visualization capabilities

## Installation

```bash
cd quadruped_parkour_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
import quadruped_parkour_env

# Create environment
env = gym.make('QuadrupedParkour-v0')

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
python test_parkour.py
```

Run comprehensive tests with visualization and performance analysis.

## Requirements

- gymnasium>=0.28.0
- mujoco>=2.3.0
- numpy>=1.21.0
- scipy>=1.7.0
- matplotlib>=3.5.0 (optional)

## Environment Details

- **Action Space**: 16 continuous torque controls for leg joints (±80Nm hip, ±60Nm knee, ±40Nm ankle)
- **Observation Space**: 95-dimensional state including joint positions/velocities, body pose, foot contacts, lidar readings, and obstacle information
- **Episode Length**: 6000 steps (60 seconds at 100Hz control frequency)
- **Course Features**: 12 distinct obstacle types with dynamic terrain and weather effects

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025
