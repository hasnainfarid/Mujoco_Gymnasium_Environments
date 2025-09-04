
<img width="1710" height="905" alt="Dance" src="https://github.com/user-attachments/assets/f253d71d-9a62-4f26-94be-78d9f79172f7" />



# Humanoid Dancing Environment

A MuJoCo-based humanoid robot dancing environment for reinforcement learning research. The environment simulates a humanoid robot learning to perform various dance moves, follow rhythm patterns, maintain balance, and execute choreographed sequences with style.

## Features

- **Rhythmic dance simulation** with 120 BPM beat synchronization, 10 different dance move types from basic steps to complex moves like pirouettes and breakdancing, and dynamic scoring system based on rhythm, style, and creativity
- **Advanced combo system** that builds combos by staying on beat and performing moves successfully, with virtual crowd simulation that reacts to performance quality
- **Comprehensive dance moves** including Basic Step, Spin, Jump, Robot Wave, Hip Hop Bounce, Freeze, Salsa Basic, Moonwalk, Breakdance Toprock, and Ballet Pirouette with varying difficulty levels
- **Visual effects and atmosphere** featuring disco ball, spotlights, stage effects, and immersive dance environment with performance metrics tracking
- **Performance analytics** including perfect move tracking, energy usage monitoring, creativity scores, and crowd excitement levels

## Installation

```bash
cd humanoid_dancing_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from humanoid_dancing_env import HumanoidDancingEnv

# Create environment
env = HumanoidDancingEnv(render_mode='human')

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
python test_dancing.py
```

Run comprehensive tests with visualization and dance performance analysis.

## Requirements

- gymnasium>=0.28.0
- mujoco>=2.3.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0 (optional)

## Environment Details

- **Action Space**: 27 continuous torque controls for humanoid joints
- **Observation Space**: 85-dimensional state including joint positions/velocities, robot pose, rhythm information, and performance metrics
- **Episode Length**: 2000 steps (20 seconds at 100Hz control frequency)
- **Dance Moves**: 10 different move types with varying difficulty levels and scoring

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025
