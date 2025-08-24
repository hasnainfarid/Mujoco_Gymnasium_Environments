# MuJoCo Gymnasium Environments

A collection of custom reinforcement learning environments built with [Gymnasium](https://gymnasium.farama.org/) and powered by [MuJoCo](https://mujoco.org/). These environments are designed for experimenting with physics-based control, robotics, and continuous action spaces.

---

## Overview

This repository provides physics simulation environments using MuJoCo. Each environment is:

* Compatible with the latest Gymnasium API
* Based on realistic physics simulations
* Easy to extend for research, projects, or teaching

---

## Features

* MuJoCo-based continuous control tasks
* Modular, self-contained environments
* Visualization with the MuJoCo viewer
* Example training and testing scripts

---

## Installation

```bash
# Clone this repo
gh repo clone hasnainfarid/Mujoco_Gymnasium_Environments
cd mujoco-gymnasium-envs

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
import gymnasium as gym
from mujoco_envs import ExampleMujocoEnv

# Create environment
env = ExampleMujocoEnv(render_mode="human")
obs, info = env.reset()

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

---

## Requirements

* Python 3.8+
* gymnasium>=0.28.0
* mujoco>=3.0.0
* numpy>=1.19.0

Additional dependencies may be listed in each environment’s `requirements.txt`.

---

## Contributing

You can:

* Add new MuJoCo tasks
* Modify existing environments
* Share examples with RL libraries (e.g., Stable-Baselines3, RLlib)

---

## License

MIT License – see `LICENSE` for details.

---

## Author

Hasnain Fareed – 2025

