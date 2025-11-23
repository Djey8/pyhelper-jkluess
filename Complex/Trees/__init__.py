"""
Trees Package - Complex Data Structures

A tree is a connected, acyclic undirected graph.
Trees are a fundamental data structure used in many applications:
- Hierarchical file systems
- Compilers and parsers
- Search trees
- Data compression
- etc.

This package provides implementations of tree data structures.
"""

try:
    from Complex.Trees.tree import Tree, TreeNode
except ImportError:
    from tree import Tree, TreeNode

__all__ = ['Tree', 'TreeNode']
__version__ = '0.1.0'
