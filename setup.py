from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyhelper-fom",  # Package name on PyPI (change if needed)
    version="0.1.0",
    author="Jannis Kluess",
    author_email="janniskluess@yahoo.de",
    description="A collection of Python data structures for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyhelper",  # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=[
        # Add any dependencies your package needs
        # For example: "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)
