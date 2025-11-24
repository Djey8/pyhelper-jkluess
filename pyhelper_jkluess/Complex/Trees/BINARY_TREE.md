# Binary Tree

A **Binary Tree** is a specialized tree data structure where each node has at most two children, referred to as the **left child** and **right child**. This implementation inherits from the `Tree` class and adds binary-specific constraints and functionality.

## Class Hierarchy

```
Node (base class)
  └── BinaryNode (extends Node)

Tree (base class)
  └── BinaryTree (extends Tree)
```

## Key Features

- **Inheritance**: BinaryTree extends Tree, gaining all standard tree operations
- **Binary Constraint**: Enforces maximum 2 children per node
- **Left/Right Properties**: BinaryNode provides `left` and `right` properties while internally using the `children` list
- **Binary-Specific Traversals**: Pre-order, In-order, Post-order traversals
- **Tree Properties**: Checks for complete, perfect, and balanced trees

## Definitions

### Binary Tree Types

#### 1. Complete Binary Tree (Vollständiger Binärbaum)
Every node has either 0 or exactly 2 children.

```
Example:
       1
      / \
     2   3
    / \
   4   5
```

Check with: `tree.is_complete()`

#### 2. Perfect Binary Tree (Perfekter Binärbaum)
A complete binary tree where all leaves are at the same level. Has `2^k - 1` nodes where `k = height + 1`.

```
Example:
         1
       /   \
      2     3
     / \   / \
    4   5 6   7
```

Check with: `tree.is_perfect()`

#### 3. Balanced Binary Tree (Balancierter Binärbaum)
Height difference between any two leaves is at most 1.

```
Example (balanced):
       1
      / \
     2   3
    / \
   4   5

Example (not balanced):
     1
    /
   2
  /
 3
```

Check with: `tree.is_balanced()`

## Tree Sorting Algorithm (Baum-Sortier-Algorithmus)

### Prerequisites
Given: n valued nodes

### Algorithm

1. The position of a first node with value k₀ is established.

2. When a second node with value k₁ is added, its value is compared with k₀ and it is:
   - placed **left** of k₀, if k₁ < k₀
   - placed **right** of k₀, if k₁ > k₀
   - **ignored**, if k₁ = k₀

3. A new node is then compared step-by-step, starting at k₀, with all subsequent nodes and passed along until it can be placed in a free position.

4. Reading the resulting tree is done **In-Order**.

### Note
To be sortable, a total order must be given for the node values, i.e., for all node values, the relationships <, =, > between any two values must be unambiguously defined.

## Traversal Orders

### Pre-Order (Präorder)
Visit order: **Root → Left → Right**

```python
tree.traverse_preorder()  # [1, 2, 4, 5, 3]
```

**Use cases**: 
- Creating a copy of the tree
- Prefix notation of expressions
- Serializing tree structure

### In-Order (Inorder)
Visit order: **Left → Root → Right**

```python
tree.traverse_inorder()  # [4, 2, 5, 1, 3]
```

**Use cases**:
- Binary Search Trees (gives sorted order)
- Infix notation of expressions

### Post-Order (Postorder)
Visit order: **Left → Right → Root**

```python
tree.traverse_postorder()  # [4, 5, 2, 3, 1]
```

**Use cases**:
- Deleting the tree (delete children before parent)
- Postfix notation of expressions
- Computing directory sizes

### Level-Order (Breitensuche)
Visit nodes level by level (breadth-first)

```python
tree.traverse_levelorder()  # [1, 2, 3, 4, 5]
```

**Use cases**:
- Finding shortest path
- Level-by-level processing
- Serialization for certain applications

## Import/Export Features

BinaryTree inherits all import/export methods from the Tree class. These allow converting between different representations.

### Nested Structure (Recommended for Binary Trees)

Best format for binary trees because it preserves left/right order and handles duplicate values:

```python
# Create from nested structure
tree = BinaryTree.from_nested_structure((1, (2, (4,), (5,)), (3, (6,), (7,))))

# Export to nested structure
structure = tree.to_nested_structure()
# Returns: (1, (2, (4,), (5,)), (3, (6,), (7,)))
```

### Adjacency Matrix/List

Works but may not preserve exact left/right order after round-trip conversion:

```python
# Export to adjacency matrix
matrix = tree.get_adjacency_matrix()
labels = tree.get_node_labels()

# Recreate from matrix
tree2 = BinaryTree.from_adjacency_matrix(matrix, labels)
# Note: Same tree structure, but left/right might swap
```

### Expression Trees with Duplicate Values

Use nested structure for trees with duplicate values (like mathematical expressions):

```python
# Expression: (3 + 5) * 3
expr_tree = BinaryTree.from_nested_structure(
    ('*', ('+', (3,), (5,)), (3,))  # Note: two nodes with value 3
)
```

### Important Notes

⚠️ **Adjacency matrix/list limitations:**
- May not preserve left vs right child positions after conversion
- Cannot distinguish between duplicate values at different positions
- Best for unique-valued trees where structure matters more than exact left/right placement

✅ **Nested structure advantages:**
- Preserves exact left/right order: `(root, (left_subtree,), (right_subtree,))`
- Handles duplicate values correctly
- Most intuitive for binary trees
- Use `()` or `None` for missing children

## Usage Examples

### Creating a Binary Tree

```python
from pyhelper_jkluess.Complex.Trees import BinaryTree

# Create with root
tree = BinaryTree(1)

# Insert children
left = tree.insert_left(tree.root, 2)
right = tree.insert_right(tree.root, 3)

# Continue building
tree.insert_left(left, 4)
tree.insert_right(left, 5)
```

### Using Inherited Tree Methods

All Tree class methods are available:

```python
# Get node count (inherited from Tree)
count = tree.get_node_count()

# Get height (overridden for binary specifics)
height = tree.get_height()

# Check if empty (inherited)
is_empty = tree.is_empty()

# Print tree structure (inherited)
tree.print_tree()
```

### Binary-Specific Operations

```python
# Traversals
preorder = tree.traverse_preorder()    # [1, 2, 4, 5, 3]
inorder = tree.traverse_inorder()      # [4, 2, 5, 1, 3]
postorder = tree.traverse_postorder()  # [4, 5, 2, 3, 1]
levelorder = tree.traverse_levelorder()  # [1, 2, 3, 4, 5]

# Properties
is_complete = tree.is_complete()
is_perfect = tree.is_perfect()
is_balanced = tree.is_balanced()
leaf_count = tree.get_leaf_count()
```

### Expression Tree Example

Binary trees are commonly used to represent mathematical expressions:

```python
# Build expression tree for: (4 + 5) * (2 - 1)
expr_tree = BinaryTree('*')
plus = expr_tree.insert_left(expr_tree.root, '+')
minus = expr_tree.insert_right(expr_tree.root, '-')
expr_tree.insert_left(plus, 4)
expr_tree.insert_right(plus, 5)
expr_tree.insert_left(minus, 2)
expr_tree.insert_right(minus, 1)

# Different notations via traversals
prefix = expr_tree.traverse_preorder()   # ['*', '+', 4, 5, '-', 2, 1]
infix = expr_tree.traverse_inorder()     # [4, '+', 5, '*', 2, '-', 1]
postfix = expr_tree.traverse_postorder() # [4, 5, '+', 2, 1, '-', '*']
```

### Binary Constraint Enforcement

The binary tree automatically enforces the 2-child limit:

```python
tree = BinaryTree(1)
tree.insert_left(tree.root, 2)
tree.insert_right(tree.root, 3)

# This will raise ValueError:
tree.insert_left(tree.root, 4)  # ValueError: already has a left child
```

## BinaryNode vs Node

`BinaryNode` extends `Node` and provides:

### Additional Properties
- `left`: Property to access/set left child (actually `children[0]`)
- `right`: Property to access/set right child (actually `children[1]`)
- `has_left_child()`: Check if left child exists
- `has_right_child()`: Check if right child exists
- `has_both_children()`: Check if both children exist
- `children_count()`: Returns 0, 1, or 2

### Inherited from Node
- `data`: The node's data
- `parent`: Reference to parent node
- `children`: List of children (max 2 for binary nodes)
- `is_leaf()`: Check if node has no children
- `is_inner_node()`: Check if node has children
- `add_child()`: Add a child (validates binary constraint)
- `remove_child()`: Remove a child

## API Reference

### BinaryTree Class

#### Constructor
```python
BinaryTree(root_data=None)
```
Creates a binary tree, optionally with root data.

#### Methods

**Insertion:**
- `insert_left(parent, data)`: Insert left child
- `insert_right(parent, data)`: Insert right child
- `add_child(parent, data)`: Add child with validation (use insert_left/right instead)
- `set_root(data)`: Set/replace root

**Traversals:**
- `traverse_preorder(node=None)`: Pre-order traversal (Root → Left → Right)
- `traverse_inorder(node=None)`: In-order traversal (Left → Root → Right)
- `traverse_postorder(node=None)`: Post-order traversal (Left → Right → Root)
- `traverse_levelorder()`: Level-order traversal (breadth-first)

**Binary Properties:**
- `get_height(node=None)`: Get tree/subtree height
- `get_leaf_count()`: Count leaf nodes
- `is_complete()`: Check if complete binary tree
- `is_perfect()`: Check if perfect binary tree
- `is_balanced()`: Check if balanced binary tree

**Import/Export (Inherited from Tree):**
- `from_adjacency_matrix(matrix, labels)`: Create from matrix (classmethod)
- `from_adjacency_list(adj_list, root)`: Create from list (classmethod)
- `from_nested_structure(structure)`: Create from nested tuples/lists (classmethod)
- `get_adjacency_matrix()`: Export to matrix
- `get_adjacency_list()`: Export to adjacency list
- `to_nested_structure()`: Export to nested structure (handles duplicates)
- `get_node_labels()`: Get labels in matrix order for reconstruction

**Other Inherited Methods:**
- `get_node_count()`: Count all nodes
- `is_empty()`: Check if tree is empty
- `print_tree()`: Print visual tree structure
- `find_node(data)`: Find node by data
- `get_depth(node)`: Get node depth
- `get_ancestors(node)`: Get all ancestors
- `get_descendants(node)`: Get all descendants
- Plus all other Tree methods

### BinaryNode Class

#### Constructor
```python
BinaryNode(data)
```
Creates a binary tree node.

#### Properties
- `left`: Get/set left child
- `right`: Get/set right child
- `data`: Node data (inherited)
- `parent`: Parent node (inherited)
- `children`: List of children (inherited, max 2)

#### Methods
- `has_left_child()`: Check if left child exists
- `has_right_child()`: Check if right child exists
- `has_both_children()`: Check if both children exist
- `children_count()`: Count children (0, 1, or 2)
- `is_leaf()`: Check if leaf node (inherited)
- `is_inner_node()`: Check if inner node (inherited)
- `add_child(child)`: Add child with validation (inherited, overridden)

## Mathematical Properties

### Perfect Binary Tree Node Count
A perfect binary tree of height `h` has:
- **Nodes**: `n = 2^(h+1) - 1`
- **Leaves**: `2^h`
- **Inner nodes**: `2^h - 1`

### Complete Binary Tree
For a complete binary tree:
- Every level except possibly the last is completely filled
- In this implementation: every node has 0 or 2 children

### Balanced Binary Tree
For a balanced binary tree:
- The height difference between any two leaves ≤ 1
- Ensures O(log n) operations in many applications

## LCRS Conversion (Left-Child Right-Sibling)

The Tree class provides a `to_binary_tree()` method that converts any general tree into a binary tree using the **Left-Child Right-Sibling (LCRS)** representation.

### LCRS Principle

In LCRS representation:
- The **left pointer** of a node points to its **first child**
- The **right pointer** of a node points to its **next sibling**

This allows any tree (with any number of children per node) to be represented as a binary tree.

### Example

Original Tree:
```
    A
   /|\
  B C D
 /|
E F
```

LCRS Binary Tree:
```
      A
     /
    B
   / \
  E   C
   \   \
    F   D
```

Explanation:
- A's left = B (first child)
- B's right = C (sibling), C's right = D (sibling)
- B's left = E (first child)
- E's right = F (sibling)

### Usage

```python
from pyhelper_jkluess.Complex.Trees.tree import Tree

# Create general tree
tree = Tree("Root")
a = tree.add_child(tree.root, "A")
b = tree.add_child(tree.root, "B")
c = tree.add_child(tree.root, "C")
tree.add_child(a, "A1")
tree.add_child(a, "A2")

# Convert to binary tree
binary = tree.to_binary_tree()
```

### preserve_binary Parameter

The `to_binary_tree(preserve_binary=False)` method accepts an optional `preserve_binary` parameter:

**`preserve_binary=False` (default - Pure LCRS):**
- Always applies LCRS consistently
- All nodes use left=first child, right=sibling
- Guarantees consistent structure

**`preserve_binary=True` (Hybrid Mode):**
- Attempts to preserve binary structure when possible
- Nodes with ≤2 children: Try to preserve as left/right children
- Nodes with >2 children: Always use LCRS
- **Important**: If a parent uses LCRS (has >2 children), child nodes cannot fully preserve their binary structure because the right pointer is needed for siblings

### Preservation Limitations

When `preserve_binary=True`, preservation only works if the node's right pointer is available:

```python
# Example 1: Full preservation possible
tree = Tree(1)
tree.add_child(tree.root, 2)
tree.add_child(tree.root, 3)  # 2 children

binary = tree.to_binary_tree(preserve_binary=True)
# Structure: root.left=2, root.right=3 (preserved!)

# Example 2: Limited preservation
tree = Tree("Root")
a = tree.add_child(tree.root, "A")
b = tree.add_child(tree.root, "B")
c = tree.add_child(tree.root, "C")  # 3 children - must use LCRS
tree.add_child(a, "A1")
tree.add_child(a, "A2")  # A has 2 children

binary = tree.to_binary_tree(preserve_binary=True)
# Root: LCRS (A.left=A, A.right=B sibling, B.right=C sibling)
# A: Cannot preserve! A.right is used for sibling B
#    So A.left=A1, A1.right=A2 (chained as siblings)
```

### When to Use Each Mode

- **Use `preserve_binary=False`**: 
  - Need consistent LCRS structure
  - Tree has many nodes with >2 children
  - Converting for educational purposes (learning LCRS)

- **Use `preserve_binary=True`**:
  - Tree is mostly binary (most nodes have ≤2 children)
  - Want more intuitive structure where possible
  - Converting partially binary trees

### Properties

- **Node preservation**: All nodes from the original tree are present in the binary tree
- **Traversal equivalence**: Pre-order traversal of original tree matches pre-order of LCRS tree
- **Reversibility**: LCRS is a lossless transformation (can reconstruct original if needed)

## Comparison with Tree Class

| Feature | Tree | BinaryTree |
|---------|------|------------|
| Max children per node | Unlimited | 2 (left, right) |
| In-order traversal | ❌ | ✅ |
| Binary constraint | ❌ | ✅ |
| Complete/Perfect check | ❌ | ✅ |
| LCRS conversion | ✅ (to binary) | N/A |
| All other operations | ✅ | ✅ (inherited) |

## Performance

All traversals and property checks run in **O(n)** time where n is the number of nodes.

- `traverse_*()`: O(n) - visits each node once
- `is_complete()`: O(n) - checks all nodes
- `is_perfect()`: O(n) - checks all leaf depths
- `is_balanced()`: O(n) - checks all leaf depths
- `get_height()`: O(n) - visits all nodes
- `get_node_count()`: O(n) - inherited from Tree

## Related Classes

- `Tree`: Parent class providing general tree operations
- `Node`: Base node class for general trees
- `Graph`: More general graph structure (trees are special graphs)

## See Also

- [Tree README](./README.md) - General tree documentation
- [Graph README](../Graphs/README.md) - Graph data structures
- [Examples](../../../examples/binary_tree_demo.py) - Binary tree usage examples
