# Complex Data Structures

**Non-linear and probabilistic data structures** for advanced algorithms, network modeling, and hierarchical data organization.

## Overview

Complex structures go beyond linear sequences to model:
- **Networks & Relationships** (Graphs) - connections between entities
- **Hierarchies & Trees** (Trees) - parent-child relationships with guaranteed properties
- **Probabilistic Optimization** (Skip Lists) - fast sorted operations without balancing

## Graphs

**Required:** `pip install networkx matplotlib`

- **Graph** Unified class supporting all 4 graph types (recommended)
- **UndirectedGraph** - Bidirectional edges (social networks, maps)
- **DirectedGraph** - One-way edges (dependencies, workflows)
- **WeightedUndirectedGraph** - Bidirectional with costs (road networks, Dijkstra)
- **WeightedDirectedGraph** - One-way with costs (optimized routing)

**Graph Theory Features:**
- Paths and reachability (Wege und Erreichbarkeit)
- Cycle detection (Zyklen)
- Connectivity analysis (Zusammenhang)
- Adjacency matrix representation (Adjazenzmatrix)
- Shortest paths (Dijkstra for weighted graphs)

```python
# Using the unified Graph class (recommended)
from pyhelper_jkluess.Complex.Graphs.graph import Graph

# Create any graph type with parameters
g = Graph(directed=False, weighted=False)
g.add_edge("A", "B")
g.add_edge("B", "C")

# Path finding
path = g.find_path("A", "C")
print(g.has_cycle())
print(g.is_connected())

# Adjacency matrix
matrix = g.get_adjacency_matrix()

g.visualize()

# Or use specialized classes
from pyhelper_jkluess.Complex.Graphs.undirected_graph import UndirectedGraph
g2 = UndirectedGraph()
```

See [Graphs README](Graphs/README.md) for details.

## Trees

**Required:** `pip install networkx matplotlib`

**A Tree is a connected, acyclic graph with one root node** - the foundation for hierarchical data.

- **Tree** - 39 operations including traversals, statistics, visualization, and adjacency support
- **Node** - Individual node with parent-child relationships

**Core Properties (Mathematical Guarantees):**
- **m = n - 1**: Exactly one fewer edge than nodes (fundamental tree property)
- **Connected**: Every node reachable from root
- **Acyclic**: No cycles (tree structure enforced)
- **Unique Paths**: Exactly one path between any two nodes

**Key Features:**
- **Traversals**: Preorder, Inorder, Postorder, Level-order (BFS)
- **Metrics**: Height, depth, degree, node/edge counts
- **Classifications**: Leaves vs inner nodes, levels
- **Path Finding**: Using Lowest Common Ancestor (LCA) algorithm
- **Adjacency Support**: Import/export as matrices and lists (4 methods)
- **Visualization**: Adaptive spacing for compact small trees, spacious large trees

```python
from pyhelper_jkluess.Complex.Trees.tree import Tree

# Create tree
tree = Tree("Root")

# Add children
child_a = tree.add_child(tree.root, "A")
child_b = tree.add_child(tree.root, "B")
tree.add_child(child_a, "A1")
tree.add_child(child_a, "A2")

# Traversals
print(tree.traverse_preorder())    # ['Root', 'A', 'A1', 'A2', 'B']
print(tree.traverse_levelorder())  # ['Root', 'A', 'B', 'A1', 'A2']

# Path between nodes
path = tree.find_path(child_a, child_b)

# Visualization
tree.print_tree()
tree.visualize(root_position="top")
```

See [Trees README](Trees/README.md) for details.

## Skip Lists

Probabilistic data structures with O(log n) operations.

- **SkipList** - Key-value store (dictionaries, indexes)
- **ProbabilisticSkipList** - Sorted values (sets, priority queues)

```python
from pyhelper_jkluess.Complex.SkipLists.skiplist import SkipList

sl = SkipList(max_level=4)
sl.insert(10, "value")
print(sl.search(10))
```

See [Skip Lists README](SkipLists/README.md) for details.
