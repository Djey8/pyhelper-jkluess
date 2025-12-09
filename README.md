# PyHelper

A comprehensive library of data structures designed for **learning, teaching, and practical use** in computer science education and software development.

## What is this?

PyHelper provides **production-ready, well-tested implementations** of fundamental and advanced data structures, organized by complexity:

### **Basic Structures** - Linear Data Organization
- **Linked Lists** (3 types): Forward-only, bidirectional, and circular traversal patterns
  - Use when: Dynamic sizing, frequent insertions/deletions, memory efficiency

### **Complex Structures** - Non-Linear Data Organization
- **Graphs** (Unified + 4 specialized types): Network relationships and connectivity
  - Use when: Modeling relationships, pathfinding, network analysis, dependencies
- **Trees**: Hierarchical parent-child structures with guaranteed properties (m = n-1)
  - Use when: File systems, org charts, decision trees, taxonomies
- **Skip Lists** (2 types): Probabilistic balanced structures for fast sorted operations
  - Use when: Sorted data with O(log n) operations without tree balancing complexity

## Prerequisites

```bash
pip install networkx matplotlib pytest
```

## Installation

### Install from PyPI (Recommended)

```bash
pip install pyhelper-jkluess
```

After installation, import as:
```python
import pyhelper_jkluess
# or
from pyhelper_jkluess.Complex.Trees.tree import Tree
```

### Install from GitHub

```bash
pip install git+https://github.com/Djey8/pyhelper-jkluess.git
```

### Install from local directory (for development)

```bash
pip install -e .
```

### Install from local directory (regular installation)

```bash
pip install .
```

## Quick Start

### Linked List
```python
from pyhelper_jkluess.Basic.Lists.linked_list import LinkedList

ll = LinkedList()
ll.append(10)
ll.append(20)
ll.print_list()  # 10 -> 20 -> None
```

### Graph (Unified Class - Recommended)
```python
from pyhelper_jkluess.Complex.Graphs.graph import Graph

# Create any graph type with parameters
g = Graph(directed=False, weighted=True)
g.add_edge("A", "B", 10)
g.add_edge("A", "C", 2)
g.add_edge("C", "B", 1)

# Find shortest path (uses Dijkstra for weighted graphs)
path, distance = g.find_shortest_path("A", "B")
print(f"Path: {path}, Distance: {distance}")  # Path: ['A', 'C', 'B'], Distance: 3

# Export/import adjacency list
adj_list = g.get_adjacency_list()
g2 = Graph(directed=False, weighted=True, data=adj_list)

# Visualize
g.visualize()  # Opens matplotlib window
```

### Tree
```python
from pyhelper_jkluess.Complex.Trees.tree import Tree

# Create tree with root
tree = Tree("Root")

# Add children
child_a = tree.add_child(tree.root, "A")
child_b = tree.add_child(tree.root, "B")
tree.add_child(child_a, "A1")
tree.add_child(child_a, "A2")

# Print structure
tree.print_tree()

# Traversals
print(tree.traverse_preorder())    # ['Root', 'A', 'A1', 'A2', 'B']
print(tree.traverse_levelorder())  # ['Root', 'A', 'B', 'A1', 'A2']

# Statistics
stats = tree.get_statistics()
print(f"Nodes: {stats['node_count']}, Height: {stats['height']}")
```

### Skip List
```python
from pyhelper_jkluess.Complex.SkipLists.probabilisticskiplist import ProbabilisticSkipList

sl = ProbabilisticSkipList()
sl.add(10)
sl.add(20)
print(sl.find(10))  # 10
```

## Learning & Demos

**New to data structures?** Start with the comprehensive demo files:

```bash
python demos/linked_lists_demo.py  # Start here: basic linear structures
python demos/tree_demo.py          # Hierarchical data
python demos/binary_tree_demo.py   # Binary trees and BSTs
python demos/heap_demo.py          # Priority queues and heap sort
python demos/skip_lists_demo.py    # Probabilistic balanced structures
python demos/graph_demo.py         # Network analysis and algorithms
```

Each demo is a **complete tutorial** with 5-14 examples, explanations, and real-world use cases. See **[demos/README.md](demos/README.md)** for the full learning path.

## Project Structure

```
pyhelper-jkluess/          # Repository root
├── demos/                # 📚 6 comprehensive tutorial files (START HERE!)
├── pyhelper_jkluess/     # Main package (import as: import pyhelper_jkluess)
│   ├── Basic/            # Linear structures: Lists (Linked, Double, Circular)
│   │   └── Lists/        # Production-ready list implementations
│   └── Complex/          # Non-linear & advanced structures
│       ├── Graphs/       # Network structures (Unified Graph + 4 types)
│       ├── Trees/        # Hierarchical structures (Tree, Node)
│       └── SkipLists/    # Probabilistic structures (2 types)
└── tests/                # 747 comprehensive tests
```

## Documentation by Data Structure Category

### Linear Structures
- **[Basic Lists](pyhelper_jkluess/Basic/README.md)** - LinkedList, DoubleLinkedList, CircularLinkedList
  - Forward-only, bidirectional, and circular traversal patterns
  - When to use: Dynamic arrays, LRU caches, round-robin scheduling

### Non-Linear Structures
- **[Graphs](pyhelper_jkluess/Complex/Graphs/README.md)** - Unified Graph + 4 specialized types
  - **Unified Architecture**: Single `Graph` class adapts to all 4 types (64% code reduction)
  - **Graph Theory**: Paths, cycles, connectivity, shortest paths (BFS/Dijkstra)
  - **Representations**: Adjacency matrices and lists (import/export)
  - When to use: Network modeling, dependencies, social networks, routing

- **[Trees](pyhelper_jkluess/Complex/Trees/README.md)** - Tree with 39 operations
  - **Properties**: m = n-1 edges, connected, acyclic, unique paths between nodes
  - **Traversals**: Preorder, inorder, postorder, level-order
  - **Features**: Depth/height, ancestors/descendants, adjacency matrix/list support
  - When to use: Hierarchies, file systems, decision trees, taxonomies

- **[Skip Lists](pyhelper_jkluess/Complex/SkipLists/README.md)** - Deterministic & Probabilistic
  - **Performance**: O(log n) operations with probabilistic balancing
  - **Types**: Key-value store (SkipList) and sorted set (ProbabilisticSkipList)
  - When to use: Sorted data without complex tree balancing

## Testing

```bash
pytest tests/ -v  # 747 comprehensive tests
```

## Contributing

We use **automated semantic versioning** with conventional commits. See:
- **[Quick Start CI/CD Guide](QUICKSTART_CI.md)** - Fast introduction to contributing
- **[Development Guide](DEVELOPMENT.md)** - Detailed CI/CD and versioning documentation

### Quick Contribution Guide

1. **Fork and clone** the repository
2. **Create a feature branch** from `develop`
3. **Use conventional commits**:
   - `feat:` for new features (minor version bump)
   - `fix:` for bug fixes (patch version bump)
   - `docs:` for documentation
4. **Push and create PR** to `develop`
5. **Merge to `main`** triggers automatic release and PyPI publish

Example:
```bash
git checkout -b feature/add-hash-table develop
git commit -m "feat(structures): add hash table implementation"
git commit -m "test(structures): add hash table tests"
git push origin feature/add-hash-table
```

## License

MIT License
