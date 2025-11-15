"""
Setup script for Dogs vs Cats Classification Project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('=')
        ]

setup(
    name="dogs-vs-cats",
    version="1.0.0",
    description="Deep learning project for classifying images as dogs or cats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/dogs-vs-cats",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "dogs-vs-cats-train=src.train:main",
            "dogs-vs-cats-predict=src.predict:main",
            "dogs-vs-cats-compare=src.compare_models:main",
            "dogs-vs-cats-validate=src.validate:main",
        ],
    },
)

