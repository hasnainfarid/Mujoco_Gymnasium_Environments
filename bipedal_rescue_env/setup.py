"""
Setup script for Bipedal Rescue Environment

Author: Hasnain Fareed
Year: 2025
License: MIT
"""

from setuptools import setup, find_packages

setup(
    name='bipedal-rescue-env',
    version='1.0.0',
    author='Hasnain Fareed',
    author_email='hasnain.fareed@example.com',
    description='A MuJoCo-based bipedal robot disaster rescue environment for Gymnasium',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hasnainfar/bipedal-rescue-env',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'bipedal_rescue_env': [
            'assets/*.xml',
        ],
    },
    install_requires=[
        'gymnasium>=0.29.0',
        'numpy>=1.21.0',
        'mujoco>=2.3.0',
        'pygame>=2.1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'isort>=5.0',
            'flake8>=4.0',
        ],
    },
    python_requires='>=3.8',
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
        'Topic :: Games/Entertainment :: Simulation',
    ],
    entry_points={
        'gymnasium.envs': [
            'BipedalRescue-v0 = bipedal_rescue_env:register_env',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/hasnainfar/bipedal-rescue-env/issues',
        'Source': 'https://github.com/hasnainfar/bipedal-rescue-env',
    },
)
