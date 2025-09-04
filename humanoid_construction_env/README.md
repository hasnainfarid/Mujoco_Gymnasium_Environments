# Humanoid Construction Environment

A sophisticated MuJoCo-based 3D construction simulation environment for reinforcement learning, featuring a 27-DOF humanoid robot performing complex construction tasks.

## Features

- **27-DOF humanoid robot** with enhanced dexterity for construction tasks, dual gripper hands for precision manipulation, safety equipment integration, and enhanced strength for lifting heavy materials
- **Comprehensive construction site** featuring fully operational tower crane with rotation and arm controls, various building blocks (small, medium, large), material storage areas, and designated foundation zones
- **Multiple task types** including precision block stacking, crane operation for heavy material movement, material transportation across site, and multi-step building projects
- **Dynamic environmental factors** with weather system (wind, rain, temperature), safety hazards and danger zones, day/night cycle with variable lighting, and ground condition variations
- **Advanced safety protocols** including safety zone management, hazard detection, and compliance with construction regulations

## Installation

```bash
cd humanoid_construction_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from humanoid_construction_env import HumanoidConstructionEnv

# Create environment
env = HumanoidConstructionEnv(render_mode='human')

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
python test_construction.py
```

Run comprehensive tests with 3D visualization and construction task validation.

## Requirements

- gymnasium>=0.28.0
- mujoco>=2.3.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0 (optional)

## Environment Details

- **Action Space**: 27 continuous torque controls for humanoid joints
- **Observation Space**: 120-dimensional state including joint positions/velocities, robot pose, crane state, material positions, and environmental conditions
- **Episode Length**: 3000 steps (30 seconds at 100Hz control frequency)
- **Construction Tasks**: Block stacking, crane operation, material transport, and structure building

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025