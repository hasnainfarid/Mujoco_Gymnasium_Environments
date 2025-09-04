# Humanoid Martial Arts Environment

A sophisticated MuJoCo-based martial arts training environment for reinforcement learning research.

## Features

- **27-DOF humanoid robot** with realistic joint limits, high-fidelity physics simulation at 60Hz, and comprehensive training dojo with 12x12 meter training area, walls, and mats
- **Comprehensive martial arts techniques** including strikes (punches, palm strikes, elbow strikes), kicks (front kick, side kick, roundhouse, axe kick), defensive moves (blocks, parries, dodges), and various stances
- **Advanced training system** with multiple training dummies, breaking boards, complex combinations, kata forms, and balance training with stance transitions
- **Multi-layered reward structure** including technique rewards, form accuracy bonuses, balance maintenance, power generation, speed bonuses, energy efficiency, and combo multipliers
- **Realistic physics simulation** using MuJoCo with proper collision detection, force sensor readings, and authentic martial arts movement dynamics

## Installation

```bash
cd humanoid_martial_arts_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from humanoid_martial_arts_env import HumanoidMartialArtsEnv

# Create environment
env = HumanoidMartialArtsEnv(render_mode='human')

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
python test_martial_arts.py
```

Run comprehensive tests with visualization and martial arts technique analysis.

## Requirements

- gymnasium>=0.28.0
- mujoco>=2.3.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0 (optional)

## Environment Details

- **Action Space**: 27 continuous torque controls for humanoid joints
- **Observation Space**: 95-dimensional state including joint positions/velocities, robot pose, training dummy positions, force sensors, and technique metrics
- **Episode Length**: 2500 steps (25 seconds at 100Hz control frequency)
- **Martial Arts Techniques**: Strikes, kicks, defensive moves, stances, and combinations

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025