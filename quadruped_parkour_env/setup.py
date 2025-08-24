"""
Setup script for Quadruped Parkour Environment
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    """Read README.md file."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Quadruped Parkour Environment - A MuJoCo-based RL environment"

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt if it exists."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Default requirements
    return [
        'gymnasium>=0.28.0',
        'mujoco>=2.3.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0'
    ]

setup(
    name='quadruped-parkour-env',
    version='1.0.0',
    author='Hasnain Fareed',
    author_email='hasnain.fareed@example.com',
    description='A MuJoCo-based quadruped parkour environment for reinforcement learning',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/hasnainfareed/quadruped-parkour-env',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Games/Entertainment :: Simulation',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.900'
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0'
        ]
    },
    include_package_data=True,
    package_data={
        'quadruped_parkour_env': [
            'assets/*.xml',
            'assets/*.png',
            'assets/*.jpg'
        ]
    },
    entry_points={
        'console_scripts': [
            'test-parkour=quadruped_parkour_env.test_parkour:main',
        ],
    },
    keywords='reinforcement-learning mujoco quadruped robotics parkour simulation',
    project_urls={
        'Bug Reports': 'https://github.com/hasnainfareed/quadruped-parkour-env/issues',
        'Source': 'https://github.com/hasnainfareed/quadruped-parkour-env',
        'Documentation': 'https://github.com/hasnainfareed/quadruped-parkour-env/wiki',
    },
)
