# Trees

Tree data structure implementation with comprehensive operations and visualization support.

**Required:** `pip install networkx matplotlib`

## Overview

A **Tree** is a connected, acyclic undirected graph with one distinguished node called the **root**.

### Properties of a Tree

- **m = n - 1**: For m edges and n nodes, a tree always has exactly one fewer edge than nodes
- **Connected**: There is exactly one path between any two nodes
- **Acyclic**: No cycles exist in a tree
- **Unique Path**: Each pair of nodes is connected by exactly one simple path
- **Hierarchical**: Nodes are organized in parent-child relationships

### Key Terminology

- **Root**: The distinguished starting node (has no parent)
- **Parent (Predecessor)**: Direct ancestor of a node
- **Child (Successor)**: Direct descendant of a node
- **Leaf**: Node without children (terminal node)
- **Inner Node**: Node with at least one child (non-terminal node)
- **Degree**: Number of children of a node
- **Depth**: Length of the path from root to a node (root has depth 0)
- **Level**: All nodes at the same depth
- **Height**: Maximum depth of any node in the tree

## Quick Start

```python
from pyhelper_jkluess.Complex.Trees.tree import Tree

# Create tree with root
tree = Tree("Root")

# Add children
child_a = tree.add_child(tree.root, "A")
child_b = tree.add_child(tree.root, "B")
child_c = tree.add_child(tree.root, "C")

# Add grandchildren
tree.add_child(child_a, "A1")
tree.add_child(child_a, "A2")
tree.add_child(child_b, "B1")

# Print tree structure
tree.print_tree()

# Output:
# └── Root
#     ├── A
#     │   ├── A1
#     │   └── A2
#     ├── B
#     │   └── B1
#     └── C
```

## Key Operations

```python
# Creation and modification
tree = Tree()                         # Empty tree
tree = Tree("Root")                   # Tree with root
root = tree.set_root("Root")          # Set/change root
child = tree.add_child(parent, "A")   # Add child
tree.is_empty()                       # Check if empty

# Node counting and properties
tree.get_node_count()                 # Total nodes (n)
tree.get_edge_count()                 # Total edges (m)
tree.verify_tree_property()           # Check m = n - 1
tree.get_depth(node)                  # Depth of node
tree.get_degree(node)                 # Number of children of node
tree.get_height()                     # Height of tree

# Navigation and levels
tree.get_level(0)                     # Nodes at level 0 (root)
tree.get_level(1)                     # All children of root
tree.get_all_levels()                 # Dict: {depth: [nodes]}

# Node classification
tree.get_leaves()                     # All leaf nodes
tree.get_inner_nodes()                # All inner nodes

# Tree traversals
tree.traverse_preorder()              # Root → Children (DFS)
tree.traverse_inorder()               # First child → Root → Others
tree.traverse_postorder()             # Children → Root (DFS)
tree.traverse_levelorder()            # Level-by-level (BFS)

# Memory-efficient generators (yield nodes one at a time)
for node in tree.iter_preorder():     # Generator version
    print(node.data)
    if node.data == "target":
        break                         # Early stopping without building full list

# Search and paths
tree.find_node("A")                   # Find node by data
tree.get_ancestors(node)              # Get all ancestors
tree.get_descendants(node)            # Get all descendants
tree.find_path(node1, node2)          # Unique path between nodes

# Graph theory properties
tree.is_connected()                   # Always True for valid tree
tree.has_cycle()                      # Always False for valid tree
tree.is_acyclic()                     # Always True for valid tree

# Statistics and analysis
tree.get_statistics()                 # Dict with all metrics
# Returns: node_count, edge_count, height, leaf_count, 
#          inner_node_count, satisfies_tree_property, 
#          is_connected, is_acyclic

# Import/Export Representations
matrix = tree.get_adjacency_matrix()      # Export to matrix
labels = tree.get_node_labels()            # Get labels for matrix reconstruction
adj_list = tree.get_adjacency_list()      # Export to adjacency list (unique values)
nested = tree.to_nested_structure()       # Export to nested structure (handles duplicates)

tree2 = Tree.from_adjacency_matrix(matrix, labels)  # Import from matrix
tree3 = Tree.from_adjacency_list(adj_list, root)    # Import from list
tree4 = Tree.from_nested_structure(nested)          # Import from nested structure

# Visualization
tree.print_tree()                     # ASCII tree structure
tree.visualize(title="My Tree", root_position="top")
# root_position: "top", "bottom", "left", "right"
```

## Creating Trees from Adjacency Representations

### From Adjacency Matrix

An adjacency matrix is a 2D array where `matrix[i][j] = 1` if node `i` is the parent of node `j`.

```python
from pyhelper_jkluess.Complex.Trees.tree import Tree

# Define adjacency matrix
# Structure: 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
matrix = [
    [0, 1, 1, 0, 0],  # Node 0 (root) → nodes 1, 2
    [0, 0, 0, 1, 1],  # Node 1 → nodes 3, 4
    [0, 0, 0, 0, 0],  # Node 2 (leaf)
    [0, 0, 0, 0, 0],  # Node 3 (leaf)
    [0, 0, 0, 0, 0]   # Node 4 (leaf)
]

# Optional: provide custom labels
labels = ['Root', 'A', 'B', 'A1', 'A2']
tree = Tree.from_adjacency_matrix(matrix, labels)

tree.print_tree()
# └── Root
#     ├── A
#     │   ├── A1
#     │   └── A2
#     └── B

# Export back to matrix
exported_matrix = tree.get_adjacency_matrix()
print(exported_matrix)  # Same as original matrix
```

### From Adjacency List

An adjacency list is a dictionary mapping each node to its list of children.

```python
# Define adjacency list
adj_list = {
    'Root': ['A', 'B', 'C'],
    'A': ['A1', 'A2'],
    'B': ['B1'],
    'C': [],
    'A1': [],
    'A2': [],
    'B1': []
}

tree = Tree.from_adjacency_list(adj_list, root='Root')

tree.print_tree()
# └── Root
#     ├── A
#     │   ├── A1
#     │   └── A2
#     ├── B
#     │   └── B1
#     └── C

# Export back to adjacency list
exported_list = tree.get_adjacency_list()
print(exported_list)  # Same as original
```

### From Nested Structure (For Duplicate Values)

When you need multiple nodes with the same value (e.g., math expression trees), use nested structure:

```python
# Math expression: (3 + 4) * 5 + 2 * 3
# Tree structure with duplicate operators:
#        +
#       / \
#      *   *
#     / \ / \
#    +  5 2  3
#   / \
#  3   4

structure = ('+', [
    ('*', [
        ('+', [3, 4]),
        5
    ]),
    ('*', [2, 3])
])

math_tree = Tree.from_nested_structure(structure)
math_tree.print_tree()
# └── +
#     ├── *
#     │   ├── +
#     │   │   ├── 3
#     │   │   └── 4
#     │   └── 5
#     └── *
#         ├── 2
#         └── 3

# Both * nodes and both 3 nodes are distinct despite having the same value

# Export back to nested structure
exported = math_tree.to_nested_structure()
# ('+', [('*', [('+', [3, 4]), 5]), ('*', [2, 3])])

# Can recreate exact tree
math_tree2 = Tree.from_nested_structure(exported)
# Both * nodes and both 3 nodes are preserved as distinct nodes
```

### Round-trip Conversion

```python
# Create tree normally
tree = Tree("X")
tree.add_child(tree.root, "Y")
tree.add_child(tree.root, "Z")

# Export to matrix and labels
matrix = tree.get_adjacency_matrix()
labels = tree.get_node_labels()  # Get labels in same order as matrix

# Import back - trees are structurally identical
tree2 = Tree.from_adjacency_matrix(matrix, labels)

# Verify they're identical
assert tree.get_node_count() == tree2.get_node_count()
assert tree.traverse_levelorder() == tree2.traverse_levelorder()
```

## Node Class

Individual tree nodes can be manipulated directly:

```python
from pyhelper_jkluess.Complex.Trees.tree import Node

# Node creation and manipulation
node = Node("MyData")
child = Node("ChildData")
node.add_child(child)
node.remove_child(child)

# Node properties
node.is_leaf()                    # Has no children?
node.is_inner_node()              # Has children?
node.degree()                     # Number of children

# Node attributes
node.data                         # Stored data
node.parent                       # Parent node
node.children                     # List of children
```

## Example Usage

### File System Tree

```python
from pyhelper_jkluess.Complex.Trees.tree import Tree

fs = Tree("/")
home = fs.add_child(fs.root, "home")
user = fs.add_child(home, "user")
fs.add_child(user, "file.txt")

fs.print_tree()
# └── /
#     └── home
#         └── user
#             └── file.txt

stats = fs.get_statistics()
print(f"Files: {stats['leaf_count']}, Depth: {stats['height']}")
```

### Organization Chart

```python
from pyhelper_jkluess.Complex.Trees.tree import Tree

org = Tree("CEO")
cto = org.add_child(org.root, "CTO")
cfo = org.add_child(org.root, "CFO")

eng = org.add_child(cto, "Eng Lead")
org.add_child(eng, "Dev 1")
org.add_child(eng, "Dev 2")
org.add_child(cfo, "Accountant")

# Analysis
print(f"Total: {org.get_node_count()}")
print(f"Levels: {org.get_height()}")
print(f"C-Level: {[n.data for n in org.get_level(1)]}")
print(f"ICs: {[n.data for n in org.get_leaves()]}")
```

## All Available Operations

### **Tree Class Operations (41 total)**

**Creation & Basic (4):**
- `Tree(root_data=None)` - Create tree
- `is_empty()` - Check if empty
- `set_root(data)` - Set/change root
- `add_child(parent, child_data)` - Add child

**Import/Export (7):**
- `from_adjacency_matrix(matrix, labels)` - Create from matrix (classmethod)
- `from_adjacency_list(adj_list, root)` - Create from list (classmethod)
- `from_nested_structure(structure)` - Create from nested tuples/lists (classmethod)
- `get_adjacency_matrix()` - Export to matrix
- `get_adjacency_list()` - Export to adjacency list (unique values only)
- `to_nested_structure()` - Export to nested structure (handles duplicates)
- `get_node_labels()` - Get labels in matrix order for reconstruction

**Node Counting & Properties (4):**
- `get_node_count()` - Total nodes (n)
- `get_edge_count()` - Total edges (m)
- `get_degree(node)` - Number of children
- `verify_tree_property()` - Check m = n - 1

**Depth & Height (2):**
- `get_depth(node)` - Node depth
- `get_height()` - Tree height

**Level Operations (2):**
- `get_level(depth)` - Nodes at level
- `get_all_levels()` - All levels dict

**Classification (2):**
- `get_leaves()` - All leaves
- `get_inner_nodes()` - All inner nodes

**Traversals (8):**
- `traverse_preorder()` - Root → Children (returns list)
- `traverse_inorder()` - First child → Root → Others (returns list)
- `traverse_postorder()` - Children → Root (returns list)
- `traverse_levelorder()` - Level-by-level (returns list)
- `iter_preorder()` - Generator: Root → Children (memory-efficient)
- `iter_inorder()` - Generator: First child → Root → Others
- `iter_postorder()` - Generator: Children → Root
- `iter_levelorder()` - Generator: Level-by-level

**Search & Paths (4):**
- `find_node(data)` - Find by data
- `get_ancestors(node)` - Parent to root
- `get_descendants(node)` - All descendants
- `find_path(from, to)` - Unique path

**Graph Theory (3):**
- `is_connected()` - Check connected
- `has_cycle()` - Check for cycles
- `is_acyclic()` - Check acyclic

**Statistics & Visualization (4):**
- `get_statistics()` - All metrics dict
- `print_tree()` - ASCII structure
- `visualize()` - Matplotlib graph
- `__str__() / __repr__()` - String representation

### **Node Class Operations (8 total)**

**Creation & Manipulation (3):**
- `Node(data)` - Create node
- `add_child(child)` - Add child
- `remove_child(child)` - Remove child

**Properties (3):**
- `is_leaf()` - Check if leaf
- `is_inner_node()` - Check if inner
- `degree()` - Number of children

**Attributes & String (2):**
- `data, parent, children` - Direct access
- `__str__() / __repr__()` - String representation

## Performance

| Operation | Time | Space |
|-----------|------|-------|
| Add/Remove child | O(1) / O(k) | O(1) |
| Find node | O(n) | O(h) |
| Depth/Height | O(h) / O(n) | O(h) |
| Traversals | O(n) | O(n) |
| Path finding | O(n) | O(h) |

*k = children count, h = height, n = nodes*

## Specialized Tree Implementations

This package includes several specialized tree implementations:

### 1. Binary Tree
A tree where each node has at most two children (left and right). See [BINARY_TREE.md](BINARY_TREE.md) for details.

**Key Features:**
- Left/right child properties
- Binary-specific traversals (preorder, inorder, postorder)
- Tree properties: complete, perfect, balanced
- Tree sorting algorithm (BST insertion)
- LCRS (Left-Child Right-Sibling) conversion

```python
from pyhelper_jkluess.Complex.Trees.binary_tree import BinaryTree

tree = BinaryTree(10)
tree.insert_left(tree.root, 5)
tree.insert_right(tree.root, 15)
```

### 2. Heap
A specialized binary tree that maintains the heap property (min-heap or max-heap). See [HEAP.md](HEAP.md) for details.

**Key Features:**
- Min-Heap and Max-Heap support
- Array-based efficient storage
- O(1) access to minimum/maximum
- O(log n) insertion and extraction
- Heap sort algorithm O(n log n)
- Extends BinaryTree with visualization

```python
from pyhelper_jkluess.Complex.Trees.heap import Heap, HeapType, heap_sort

# Min-Heap
heap = Heap(HeapType.MIN, [5, 3, 7, 1, 9])
min_val = heap.heap_extract()  # 1

# Heap Sort
sorted_data = heap_sort([5, 3, 7, 1, 9])  # [1, 3, 5, 7, 9]
```

### Documentation Links

- **[Binary Tree Documentation](BINARY_TREE.md)** - Complete binary tree reference
- **[Heap Documentation](HEAP.md)** - Heap data structure and heap sort
