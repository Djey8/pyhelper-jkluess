from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyhelper-jkluess",  # Package name on PyPI
    version="1.1.0",
    author="Jannis Kluess",
    author_email="janniskluess@yahoo.de",
    description="A collection of Python data structures for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Djey8/pyhelper-jkluess.git",
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
        "networkx>=2.5",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)
