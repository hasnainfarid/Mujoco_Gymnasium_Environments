"""
Setup script for Humanoid Dancing Environment

Author: Hasnain Fareed
Year: 2025
License: MIT
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="humanoid_dancing_env",
    version="1.0.0",
    author="Hasnain Fareed",
    author_email="",
    description="A MuJoCo-based humanoid dancing environment for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.28.0",
        "mujoco>=2.3.0",
        "numpy>=1.20.0",
        "pygame>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    include_package_data=True,
    package_data={
        "humanoid_dancing_env": ["assets/*.xml"],
    },
)
