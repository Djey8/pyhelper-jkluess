# PyHelper

Python implementations of fundamental data structures for learning and practical use.

## What is this?

PyHelper provides clean, well-tested implementations of common data structures:
- **Lists**: Linked, Double, Circular
- **Graphs**: Unified Graph class + specialized types (Undirected, Directed, Weighted) with visualization & graph theory operations
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
print(g.get_neighbors("A"))  # ['B']
print(g.get_edge_weight("A", "B"))  # 10
g.visualize()  # Opens matplotlib window

# Or use specialized classes
from Complex.Graphs.undirected_graph import UndirectedGraph
g2 = UndirectedGraph()
g2.add_edge("A", "B")
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
├── Basic/Lists/          # Linked list implementations
├── Complex/Graphs/       # Graph data structures
└── Complex/SkipLists/    # Skip list implementations
```

## Documentation

- [Basic Lists](Basic/Lists/README.md) - LinkedList, DoubleLinkedList, CircularLinkedList
- [Graphs](Complex/Graphs/README.md) - **Graph** (unified), UndirectedGraph, DirectedGraph, WeightedUndirectedGraph, WeightedDirectedGraph
  - Includes: Paths, cycles, connectivity, adjacency matrices, Dijkstra's algorithm
  - **NEW:** Unified `Graph` class adapts to all 4 types based on initialization
- [Skip Lists](Complex/SkipLists/README.md) - SkipList, ProbabilisticSkipList

## Testing

```bash
pytest tests/ -v  # 184 tests
```

## License

MIT License
