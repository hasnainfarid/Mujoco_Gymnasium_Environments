
<img width="1730" height="887" alt="Screenshot 2025-09-05 010119" src="https://github.com/user-attachments/assets/d2661b0c-08e3-4a62-a3e6-446112baf761" />



# Robotic Arm Assembly Environment

A MuJoCo-based gymnasium environment for industrial robotic assembly tasks with precision manipulation.

## Features

- **7-DOF industrial robot arm** with high-fidelity simulation of KUKA-style robotic arm, precise joint control, realistic torque limits, and force-sensitive manipulation capabilities
- **Multi-component assembly task** featuring complete assembly sequence with 8 different components including PCB, CPU, screws, battery, and cover with precision control requirements
- **Advanced manipulation system** with ±2mm positioning accuracy, adaptive grasp planning, compliant motion control, and integrated force/torque sensing for delicate component handling
- **Damage prevention mechanisms** with force-sensitive manipulation, collision detection, and safety protocols to prevent component damage during assembly
- **Comprehensive assembly environment** with assembly station setup, gripper tools, and realistic industrial workspace simulation

## Installation

```bash
cd robotic_arm_assembly_env
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from robotic_arm_assembly_env import RoboticArmAssemblyEnv

# Create environment
env = RoboticArmAssemblyEnv(render_mode='human')

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
python test_assembly.py
```

Run comprehensive tests with visualization and assembly task validation.

## Requirements

- gymnasium>=0.28.0
- mujoco>=2.3.0
- numpy>=1.21.0
- pygame>=2.1.0
- matplotlib>=3.5.0 (optional)

## Environment Details

- **Action Space**: 7 continuous torque controls for robot arm joints
- **Observation Space**: 45-dimensional state including joint positions/velocities, end-effector pose, component positions, and force sensor readings
- **Episode Length**: 1500 steps (15 seconds at 100Hz control frequency)
- **Assembly Components**: 8 different components with specific assembly sequence requirements

## License

MIT License - Copyright (c) 2025 Hasnain Fareed

## Author

Hasnain Fareed - 2025
