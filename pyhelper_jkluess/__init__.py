"""
pyhelper-jkluess - A collection of Python data structures
Educational implementations of fundamental and advanced data structures.
"""

__version__ = "1.1.0"
__author__ = "Jannis Kluess"

# Import modules to make them easily accessible
from . import Basic
from . import Complex

# Make lowercase aliases for convenience
basic = Basic
complex = Complex

__all__ = ['Basic', 'Complex', 'basic', 'complex']
