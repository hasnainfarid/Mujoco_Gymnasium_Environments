# Humanoid Soccer Environment

A comprehensive MuJoCo-based 3D soccer simulation environment for reinforcement learning research. This package provides a realistic humanoid robot soccer environment with advanced physics simulation, challenging gameplay mechanics, and comprehensive reward systems.

## Recent Improvements (2025)

- **Cleaner Soccer Field**: Removed unnecessary objects like wind zones, extra lighting, and complex boundary walls
- **Realistic Humanoid**: Improved proportions, better joint ranges, and enhanced foot design for soccer gameplay
- **Focused Environment**: Streamlined field design that focuses on soccer gameplay without distractions

## Features

- **25 degrees of freedom humanoid robot** with realistic proportions, improved joint limits, and advanced sensors (IMU, joint encoders, contact sensors, head-mounted camera)
- **FIFA-standard soccer field** (50m × 30m) with proper markings, realistic ball physics, goal detection, and clean design
- **Intelligent opponent system** with PID-controlled goalkeeper that adapts to ball position and velocity
- **Comprehensive reward system** including goal scoring (+10,000), ball control (+1,000), movement efficiency, posture maintenance, and energy efficiency
- **Advanced physics simulation** with MuJoCo integration, realistic collision detection, and dynamic interactions

## Installation

```bash
cd humanoid_soccer_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
import humanoid_soccer_env

# Create environment
env = gym.make('HumanoidSoccer-v0', render_mode='human')

# Reset environment
observation, info = env.reset()

# Run episode
for step in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        print(f"Final reward: {reward}")
        print(f"Goals scored: {info['episode_stats']['goals_scored']}")
        break

env.close()
```

## Testing

```bash
python test_soccer.py
```

Run comprehensive tests with visualization and performance analysis. The test script will open a 3D visualization window to showcase the improved environment.

## Environment Details

### Field Design
- Clean grass field with proper soccer markings
- Goal posts with nets
- Penalty areas and goal areas
- Center circle and center line
- Corner arcs for realistic soccer field appearance

### Humanoid Robot
- Realistic body proportions (height: ~1.5m)
- Improved joint ranges for natural movement
- Better foot design optimized for soccer
- 25 degrees of freedom for complex movements

## Requirements

- gymnasium>=0.28.0
- mujoco>=2.3.0
- numpy>=1.21.0
- scipy>=1.7.0
- matplotlib>=3.5.0 (optional)
- opencv-python>=4.5.0 (optional)

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025
