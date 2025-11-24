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
    from pyhelper_jkluess.Complex.Trees.tree import Tree, Node
except ImportError:
    try:
        from .tree import Tree, Node
    except ImportError:
        from tree import Tree, Node

# Export Node as TreeNode for consistency with package naming
TreeNode = Node

__all__ = ['Tree', 'TreeNode', 'Node']
__version__ = '0.1.0'
