"""
Setup script for Make Data Count competition package
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().splitlines()

setup(
    name="make-data-count-nlp",
    version="0.1.0",
    description="NLP solution for Make Data Count Kaggle competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/make-data-count-nlp",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, machine-learning, data-science, kaggle, competition",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/make-data-count-nlp/issues",
        "Source": "https://github.com/yourusername/make-data-count-nlp",
        "Documentation": "https://github.com/yourusername/make-data-count-nlp#readme",
    },
) 