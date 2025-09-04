"""
Setup script for Humanoid Martial Arts Environment

Author: Hasnain Fareed
Year: 2025
License: MIT
"""

from setuptools import setup, find_packages

setup(
    name='humanoid-martial-arts-env',
    version='1.0.0',
    author='Hasnain Fareed',
    author_email='',
    description='A MuJoCo-based humanoid martial arts training environment for reinforcement learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['assets/*.xml'],
    },
    install_requires=[
        'gymnasium>=0.29.0',
        'numpy>=1.24.0',
        'mujoco>=3.0.0',
        'pygame>=2.5.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    entry_points={
        'gymnasium.envs': [
            'HumanoidMartialArts-v0 = humanoid_martial_arts_env:HumanoidMartialArtsEnv',
        ]
    },
)
