#!/usr/bin/env python3
"""
Setup script for Geometric ML Framework

Provides proper package installation with entry points for CLI commands.
Supports both development and production installations.

Author: Sar Hamam
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_readme():
    """Read README.md for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Geometric ML Framework - Unifying dual transports and topological analysis"

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read development requirements
def read_dev_requirements():
    """Read development requirements from requirements-dev.txt."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements-dev.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Version management
def get_version():
    """Get version from src/__init__.py or fallback to default."""
    version_file = os.path.join(os.path.dirname(__file__), 'src', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

setup(
    # Basic package information
    name="geometric-ml",
    version=get_version(),
    author="Sar Hamam",
    author_email="sar.hamam@example.com",  # Update with actual email
    description="Geometric ML Framework - Unifying dual transports and topological analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sarhamam/geometric-ml",  # Update with actual repo

    # Package structure
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",

    # Dependencies
    install_requires=read_requirements(),
    extras_require={
        "dev": read_dev_requirements(),
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ],
        "full": read_requirements() + read_dev_requirements() + [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ]
    },

    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "geometric-ml=run_geometric_ml:main",
            "gml=run_geometric_ml:main",
            "geometric-ml-demo=run_geometric_ml:run_quick_demo",
            "gml-demo=run_geometric_ml:run_quick_demo",
        ],
    },

    # Include additional files
    include_package_data=True,
    package_data={
        "": [
            "configs/*.yaml",
            "notebooks/*.ipynb",
            "examples/*.py",
            "docs/*.md",
        ],
    },

    # Classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],

    # Keywords for discovery
    keywords=[
        "geometric-ml",
        "machine-learning",
        "geometry",
        "topology",
        "spectral-analysis",
        "dual-transport",
        "manifold-learning",
        "graph-theory",
        "laplacian",
        "mellin-transform",
        "fisher-rao"
    ],

    # Project URLs
    project_urls={
        "Documentation": "https://github.com/sarhamam/geometric-ml/docs",
        "Source": "https://github.com/sarhamam/geometric-ml",
        "Bug Reports": "https://github.com/sarhamam/geometric-ml/issues",
        "Examples": "https://github.com/sarhamam/geometric-ml/examples",
    },

    # Additional metadata
    zip_safe=False,  # For proper package data access
    platforms=["any"],

    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
    ],
)