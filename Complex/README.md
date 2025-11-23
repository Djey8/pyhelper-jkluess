# Complex Data Structures

Advanced structures: Graphs and Skip Lists.

## Graphs

**Required:** `pip install networkx matplotlib`

- **Graph** - **NEW!** Unified class supporting all 4 graph types (recommended)
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
from Complex.Graphs.graph import Graph

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
from Complex.Graphs.undirected_graph import UndirectedGraph
g2 = UndirectedGraph()
```

See [Graphs README](Graphs/README.md) for details.

## Skip Lists

Probabilistic data structures with O(log n) operations.

- **SkipList** - Key-value store (dictionaries, indexes)
- **ProbabilisticSkipList** - Sorted values (sets, priority queues)

```python
from Complex.SkipLists.skiplist import SkipList

sl = SkipList(max_level=4)
sl.insert(10, "value")
print(sl.search(10))
```

See [Skip Lists README](SkipLists/README.md) for details.
