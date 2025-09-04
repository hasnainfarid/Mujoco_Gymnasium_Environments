"""
Setup script for Humanoid Construction Environment package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    """Read README.md file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A MuJoCo-based humanoid construction environment for reinforcement learning."

# Package metadata
PACKAGE_NAME = "humanoid_construction_env"
VERSION = "1.0.0"
AUTHOR = "Hasnain Fareed"
AUTHOR_EMAIL = "hasnain.fareed@example.com"
DESCRIPTION = "MuJoCo-based humanoid construction environment for reinforcement learning"
LICENSE = "MIT"
YEAR = "2025"

# Dependencies
REQUIRED_PACKAGES = [
    "gymnasium>=0.28.0",
    "mujoco>=2.3.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pygame>=2.1.0",  # For visualization
]

OPTIONAL_PACKAGES = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'black>=21.0.0',
        'flake8>=3.9.0',
        'mypy>=0.910',
    ],
    'viz': [
        'matplotlib>=3.5.0',
        'opencv-python>=4.5.0',
        'pillow>=8.3.0',
    ]
}

# Classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    f"License :: OSI Approved :: {LICENSE} License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Games/Entertainment :: Simulation",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Keywords
KEYWORDS = [
    "reinforcement-learning",
    "mujoco",
    "robotics",
    "construction",
    "humanoid",
    "simulation",
    "gymnasium",
    "ai",
    "machine-learning",
    "building",
    "crane"
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=f"https://github.com/{AUTHOR.lower().replace(' ', '')}/{PACKAGE_NAME}",
    
    # Package configuration
    packages=find_packages(),
    package_data={
        PACKAGE_NAME: [
            "assets/*.xml",
            "assets/*.png",
            "assets/*.jpg",
            "*.md",
            "LICENSE"
        ]
    },
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=REQUIRED_PACKAGES,
    extras_require=OPTIONAL_PACKAGES,
    
    # Metadata
    classifiers=CLASSIFIERS,
    keywords=" ".join(KEYWORDS),
    
    # Entry points
    entry_points={
        'console_scripts': [
            f'{PACKAGE_NAME}-test=humanoid_construction_env.test_construction:main',
        ],
    },
    
    # Additional metadata
    project_urls={
        "Bug Reports": f"https://github.com/{AUTHOR.lower().replace(' ', '')}/{PACKAGE_NAME}/issues",
        "Source": f"https://github.com/{AUTHOR.lower().replace(' ', '')}/{PACKAGE_NAME}",
        "Documentation": f"https://github.com/{AUTHOR.lower().replace(' ', '')}/{PACKAGE_NAME}#readme",
    },
    
    zip_safe=False,
)

if __name__ == "__main__":
    print(f"Setting up {PACKAGE_NAME} v{VERSION}")
    print(f"Author: {AUTHOR}")
    print(f"License: {LICENSE}")
    print(f"Year: {YEAR}")
    print("\nTo install in development mode:")
    print("pip install -e .")
    print("\nTo install with optional dependencies:")
    print("pip install -e .[dev,viz]")
