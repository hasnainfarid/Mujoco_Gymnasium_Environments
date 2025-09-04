"""
Setup script for Robotic Arm Assembly Environment
Author: Hasnain Fareed
Year: 2025
"""

from setuptools import setup, find_packages

setup(
    name='robotic-arm-assembly-env',
    version='1.0.0',
    author='Hasnain Fareed',
    author_email='',
    description='MuJoCo-based robotic arm assembly environment for precision manipulation tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'robotic_arm_assembly_env': ['assets/*.xml'],
    },
    install_requires=[
        'gymnasium>=0.29.0',
        'numpy>=1.21.0',
        'mujoco>=2.3.0',
    ],
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
    ],
    keywords='reinforcement-learning robotics assembly mujoco gymnasium',
    license='MIT',
)
