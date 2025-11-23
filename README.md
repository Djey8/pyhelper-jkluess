# PyHelper

Python implementations of fundamental data structures for learning and practical use.

## What is this?

PyHelper provides clean, well-tested implementations of common data structures:
- **Lists**: Linked, Double, Circular
- **Graphs**: Unified Graph class + specialized types (Undirected, Directed, Weighted) with visualization & graph theory operations
- **Trees**: Hierarchical tree structures with comprehensive operations and traversals
- **Skip Lists**: Deterministic and Probabilistic

## Prerequisites

```bash
pip install networkx matplotlib pytest
```

## Installation

### Install from local directory (for development)

```bash
pip install -e .
```

### Install from local directory (regular installation)

```bash
pip install .
```

### Install from GitHub

```bash
pip install git+https://github.com/Djey8/PyHelper.git
```

### Install from PyPI

```bash
pip install pyhelper-jkluess
```

## Quick Start

### Linked List
```python
from Basic.Lists.linked_list import LinkedList

ll = LinkedList()
ll.append(10)
ll.append(20)
ll.print_list()  # 10 -> 20 -> None
```

### Graph (Unified Class - Recommended)
```python
from Complex.Graphs.graph import Graph

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
from Complex.Trees.tree import Tree

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
from Complex.SkipLists.probabilisticskiplist import ProbabilisticSkipList

sl = ProbabilisticSkipList()
sl.add(10)
sl.add(20)
print(sl.find(10))  # 10
```

## Structure

```
PyHelper/
├── Grundlegende_Datenstrukuren/  # Basic data structures (Lists)
├── Complex/Graphs/                # Graph data structures
├── Complex/Trees/                 # Tree data structures
└── Complex/SkipLists/             # Skip list implementations
```

## Documentation

- [Basic Lists](Grundlegende_Datenstrukuren/README.md) - LinkedList, DoubleLinkedList, CircularLinkedList
- [Graphs](Complex/Graphs/README.md) - **Graph** (unified), UndirectedGraph, DirectedGraph, WeightedUndirectedGraph, WeightedDirectedGraph
  - Unified `Graph` class adapts to all 4 types based on initialization
  - Shortest path algorithms: BFS (unweighted) and Dijkstra (weighted)
  - Includes: Paths, cycles, connectivity, adjacency matrices/lists
  - 64% code reduction through inheritance architecture
- [Trees](Complex/Trees/README.md) - Tree, TreeNode
  - Hierarchical tree structures with parent-child relationships
  - Properties: m = n - 1, connected, acyclic, unique paths
  - Traversals: preorder, postorder, level-order
  - Operations: depth, height, levels, ancestors, descendants, path finding
- [Skip Lists](Complex/SkipLists/README.md) - SkipList, ProbabilisticSkipList

## Testing

```bash
pytest tests/ -v  # 529 tests
```

## License

MIT License
