# Graphs

**A Graph is a collection of vertices (nodes) connected by edges** - the mathematical foundation for modeling networks and relationships.

**Required:** `pip install networkx matplotlib`

## Core Concept

Graphs model **pairwise relationships** between entities:
- **Vertices (Nodes)**: The entities (people, cities, web pages, tasks)
- **Edges (Connections)**: The relationships (friendships, roads, links, dependencies)
- **Directed vs Undirected**: One-way vs bidirectional relationships
- **Weighted vs Unweighted**: Relationships with costs/distances vs simple connections

**Real-world applications**: Social networks, maps/navigation, task scheduling, network topology, recommendation systems

## Architecture Overview

All graph classes inherit from a single, powerful **`Graph`** base class that adapts to 4 different graph types:
- **UndirectedGraph** - inherits from `Graph(directed=False, weighted=False)`
- **DirectedGraph** - inherits from `Graph(directed=True, weighted=False)`
- **WeightedUndirectedGraph** - inherits from `Graph(directed=False, weighted=True)`
- **WeightedDirectedGraph** - inherits from `Graph(directed=True, weighted=True)`


## Graph - Unified Base Class (Use Directly or via Specialized Classes)

A single, flexible `Graph` class that adapts to all 4 graph types based on initialization parameters.

```python
from pyhelper_jkluess.Complex.Graphs.graph import Graph

# Undirected unweighted graph
g1 = Graph(directed=False, weighted=False)
g1.add_edge("A", "B")
g1.add_edge("B", "C")
print(g1.is_directed)  # False
print(g1.is_weighted)  # False

# Directed unweighted graph
g2 = Graph(directed=True, weighted=False)
g2.add_edge("Task1", "Task2")
print(g2.out_degree("Task1"))  # 1

# Undirected weighted graph
g3 = Graph(directed=False, weighted=True)
g3.add_edge("Berlin", "Munich", 584)
print(g3.get_edge_weight("Berlin", "Munich"))  # 584

# Directed weighted graph
g4 = Graph(directed=True, weighted=True)
g4.add_edge("A", "B", 10)
print(g4.weighted_out_degree("A"))  # 10
```

**Key Features:**
- **Automatic Adaptation:** Methods behave according to graph type
- **All Operations:** Supports all operations from specialized classes
- **Graph Theory:** Full path/cycle/connectivity analysis
- **Visualization:** Automatic styling based on type
- **Matrix Conversion:** Bidirectional adjacency matrix support

**Use this class for:** Maximum flexibility when graph type may change or when working with multiple graph types in one project.

## Key Operations

```python
# Add/remove
g.add_vertex("A")
g.add_edge("A", "B")              # Unweighted
g.add_edge("A", "B", 10)          # Weighted graphs
g.remove_edge("A", "B")
g.remove_vertex("A")

# Query
g.has_vertex("A")
g.has_edge("A", "B")
g.get_vertices()
g.get_edges()
g.get_neighbors("A")

# Weighted graphs only
g.get_edge_weight("A", "B")       # Get edge weight
g.update_edge_weight("A", "B", 15)  # Update edge weight
g.get_weighted_neighbors("A")     # WeightedUndirected: {'B': 10, ...}, Others: [(neighbor, weight), ...]
g.weighted_degree("A")            # Sum of edge weights
g.total_weight()                  # Sum of all edge weights
g.get_weight_statistics()         # Min/max/average/total weights (specialized classes)

# DirectedGraph only
g.get_predecessors("A")           # Who points to A?
g.in_degree("A")                  # How many point to A?
g.out_degree("A")                 # How many does A point to?

# Graph theory (all graph types)
g.degree("A")                     # Total degree
g.is_simple_graph()               # Check for self-loops
g.get_degree_sequence()           # Degree sequence (specialized classes)
g.get_graph_info()                # Statistics dict (specialized classes)
g.print_graph_analysis()          # Detailed analysis (specialized classes)

# Graph class properties
g.is_directed                     # Check if directed
g.is_weighted                     # Check if weighted

# Paths and cycles (all graph types)
g.find_path("A", "B")             # Find a path between vertices (BFS)
g.find_shortest_path("A", "B")    # Find shortest path (returns (path, distance))
                                  # - Unweighted: BFS (shortest by edge count)
                                  # - Weighted: Dijkstra (shortest by total weight)
g.is_reachable("A", "B")          # Check reachability
g.path_length(path)               # Number of edges in path
g.path_weight(path)               # Sum of edge weights (weighted only)
g.has_cycle()                     # Check for cycles
g.find_cycles()                   # Find all simple cycles
g.find_all_cycles()               # Find all cycles comprehensively
g.is_acyclic()                    # Check if acyclic

# Search algorithms (all graph types)
g.dfs("A")                        # Depth-first search from A (returns list)
g.dfs("A", end="D")              # DFS stopping at D
g.bfs("A")                        # Breadth-first search from A (returns list)
g.bfs("A", end="D")              # BFS stopping at D

# Memory-efficient generators (yield vertices one at a time)
for vertex in g.iter_dfs("A"):   # Generator: DFS traversal
    print(vertex)
    if vertex == "D":
        break                     # Early stopping without building full list
for vertex in g.iter_bfs("A"):   # Generator: BFS traversal
    process(vertex)

g.dijkstra("A")                   # Dijkstra's algorithm from A (weighted only)
                                  # Returns: Dict[vertex: (distance, path)]

# Minimum spanning tree (weighted undirected only)
g.minimum_spanning_tree_kruskal() # MST using Kruskal's algorithm
g.minimum_spanning_tree_prim()    # MST using Prim's algorithm

# Connectivity (all graph types)
g.is_connected()                  # Undirected: is connected
g.is_strongly_connected()         # Directed: is strongly connected
g.get_connected_components()      # Undirected: connected components
g.get_strongly_connected_components()  # Directed: strongly connected components

# Tree properties (all graph types)
g.is_tree()                       # Check if graph is a tree
g.get_edge_count()                # Count edges (undirected counts each once)

# Matrix and list operations (all graph types)
g.get_adjacency_matrix()          # Get adjacency matrix
g.get_adjacency_list()            # Get adjacency list (for creating new graphs)
Graph.from_adjacency_matrix(matrix, vertices)  # Create from matrix

# Visualize
g.visualize(title="My Graph", figsize=(12, 8))
# Custom node positioning for visualization
my_pos = {
    'A': (0, 1),
    'B': (0, 0), 
    'C': (2, 0),
    'D': (2, 1),
    'E': (1, 3)
}
g.visualize(positions=custom_pos)  # Custom node positions
```

## UndirectedGraph - Symmetric Edges (Inherits from Graph)

Edges work both ways: A—B means A connects to B and B connects to A. Inherits all functionality from `Graph` and adds convenience methods.

```python
from pyhelper_jkluess.Complex.Graphs.undirected_graph import UndirectedGraph

g = UndirectedGraph()
g.add_edge("Alice", "Bob")
g.add_edge("Bob", "Charlie")

print(g.get_neighbors("Bob"))  # ['Alice', 'Charlie']
print(g.has_edge("Alice", "Bob"))  # True
print(g.has_edge("Bob", "Alice"))  # True (symmetric!)

# Graph theory analysis
print(g.degree("Bob"))  # 2
print(g.is_simple_graph())  # True (no self-loops)
g.print_graph_analysis()  # Detailed statistics

g.visualize(title="Social Network")
```

**Use for:** Social networks, road maps, any bidirectional relationship.

## DirectedGraph - One-Way Edges (Inherits from Graph)

Edges have direction: A→B doesn't mean B→A. Inherits all functionality from `Graph` with directed-specific methods.

```python
from pyhelper_jkluess.Complex.Graphs.directed_graph import DirectedGraph

g = DirectedGraph()
g.add_edge("Task1", "Task2")  # Task1 must complete before Task2
g.add_edge("Task2", "Task3")

# Outgoing edges (what depends on this?)
print(g.get_neighbors("Task1"))  # ['Task2']

# Incoming edges (what does this depend on?)
print(g.get_predecessors("Task2"))  # ['Task1']

# Count connections
print(g.out_degree("Task2"))  # 1 (Task2 → Task3)
print(g.in_degree("Task2"))   # 1 (Task1 → Task2)
print(g.degree("Task2"))      # 2 (total: in + out)

# Graph theory analysis
print(g.is_simple_graph())  # True (no self-loops)
g.print_graph_analysis()    # Detailed statistics

g.visualize(title="Task Dependencies")
```

**Use for:** Task dependencies, web links, workflow, anything with direction.

## WeightedUndirectedGraph - Edges with Costs (Inherits from Graph)

Like UndirectedGraph but edges have weights (distances, costs, etc.). Inherits all functionality from `Graph` with weighted analysis methods.

```python
from pyhelper_jkluess.Complex.Graphs.weighted_undirected_graph import WeightedUndirectedGraph

g = WeightedUndirectedGraph()
g.add_edge("Berlin", "Munich", 584)      # 584 km
g.add_edge("Munich", "Vienna", 434)      # 434 km
g.add_edge("Berlin", "Vienna", 680)      # 680 km

print(g.get_edge_weight("Berlin", "Munich"))  # 584
print(g.get_weighted_neighbors("Munich"))  # {'Berlin': 584, 'Vienna': 434}

# Weight statistics
stats = g.get_weight_statistics()
print(stats)  # {'min_weight': 434, 'max_weight': 680, ...}

g.visualize(title="City Network")
```

**Use for:** Road networks, flight routes, weighted social connections.

## WeightedDirectedGraph - One-Way Edges with Costs (Inherits from Graph)

Like DirectedGraph but edges have weights. Inherits all functionality from `Graph` with weighted directed-specific methods.

```python
from pyhelper_jkluess.Complex.Graphs.weighted_directed_graph import WeightedDirectedGraph

g = WeightedDirectedGraph()
g.add_edge("A", "B", 10)  # A → B costs 10
g.add_edge("B", "C", 5)   # B → C costs 5
g.add_edge("A", "C", 20)  # A → C costs 20

print(g.get_edge_weight("A", "B"))  # 10
print(g.out_degree("A"))           # 2 outgoing edges

# Get weight statistics
stats = g.get_weight_statistics()
print(stats)  # {'min_weight': 5, 'max_weight': 20, ...}

# Get weighted degree sequences
seq = g.get_weighted_degree_sequence()

g.visualize(title="Task Network")
```

**Use for:** Optimized task scheduling, network routing, weighted dependencies.

## Graph Theory Concepts

All graph implementations support fundamental graph theory operations.

### Paths and Reachability (Wege und Erreichbarkeit)

**Theory:** A path (Weg/Pfad) from vertex `v` to `v'` is a sequence of vertices v₀, v₁, ..., vₖ where:
- v₀ = v (start vertex / Anfangsknoten)
- vₖ = v' (end vertex / Endknoten)
- Edges exist between consecutive vertices
- k is the length of the path

A path is **simple** if all vertices are pairwise distinct. Vertex v' is **reachable** from v if a path exists from v to v'.

```python
# Find a path between two vertices (BFS)
path = g.find_path("A", "D")
print(path)  # ['A', 'B', 'C', 'D']

# Find shortest path (optimal for all graph types)
path, distance = g.find_shortest_path("A", "D")
print(f"Path: {path}, Distance: {distance}")
# For unweighted: uses BFS, distance = edge count
# For weighted: uses Dijkstra, distance = total weight

# Check if vertex is reachable
print(g.is_reachable("A", "D"))  # True

# Path properties
length = g.path_length(path)  # Number of edges
is_simple = g.is_simple_path(path)  # All vertices distinct?

# For weighted graphs: get path weight
weight = g.path_weight(path)  # Sum of edge weights
```

**Shortest Path Examples:**

```python
# Unweighted graph - shortest by edge count
g = Graph(directed=False, weighted=False)
g.add_edge('A', 'B')
g.add_edge('B', 'C')
g.add_edge('A', 'D')
g.add_edge('D', 'C')

path, distance = g.find_shortest_path('A', 'C')
# Returns: (['A', 'B', 'C'] or ['A', 'D', 'C'], 2)

# Weighted graph - shortest by total weight
g = Graph(directed=False, weighted=True)
g.add_edge('A', 'B', 4)
g.add_edge('A', 'C', 2)
g.add_edge('C', 'B', 1)

path, distance = g.find_shortest_path('A', 'B')
# Returns: (['A', 'C', 'B'], 3) - goes via C because 2+1 < 4
```

**Note:** The geometric position of vertices in visualizations has no mathematical significance. Two vertices must not be at the same position.

### Cycles and Acyclic Graphs (Zyklen und azyklische Graphen)

**Theory:** 
- A **cycle** (Zyklus) is a path v₀, v₁, ..., vₖ where v₀ = vₖ
- A **circle** (Kreis) is a cycle where v₀, v₁, ..., vₖ₋₁ are pairwise distinct
- Trivial cycles (length 1 or 2) are often not considered
- An **acyclic** graph (azyklisch) contains no cycles
- For directed graphs: DAG (Directed Acyclic Graph)

```python
# Check for cycles
print(g.has_cycle())  # True/False

# Find all simple cycles
cycles = g.find_cycles()
print(cycles)  # [['A', 'B', 'C'], ['D', 'E']]

# Check if acyclic (no cycles)
print(g.is_acyclic())  # True/False
```

### Connectivity (Zusammenhang)

**Undirected Graphs:**
- **Connected**: Every vertex is reachable from every other vertex
- **Connected component** (Zusammenhangskomponente): A maximal connected subgraph
- Equivalence classes with respect to "reachable from" relation

**Directed Graphs:**
- **Strongly connected**: Every vertex is reachable from every other vertex following directed edges
- **Strongly connected component**: A maximal subgraph where all vertices are mutually reachable
- Equivalence classes with respect to "mutually reachable" relation

```python
# Undirected graphs
print(g.is_connected())  # True if whole graph is connected

components = g.get_connected_components()
print(components)  # [{'A', 'B', 'C'}, {'D', 'E'}]

# Directed graphs
print(g.is_strongly_connected())  # True if all mutually reachable

components = g.get_strongly_connected_components()
print(components)  # [{'A', 'B'}, {'C', 'D', 'E'}]
```

### Tree Detection

**Theory:** A graph G with m edges and n nodes is a tree if:

**Undirected Graph:** ONE of the following conditions:
1. G is connected AND m = n - 1
2. G has no cycles AND m = n - 1  
3. There is exactly one path between every pair of nodes

**Directed Graph:** Must satisfy:
- The underlying undirected graph is a tree with a root
- There is exactly one path from the root to every other node
- Exactly one node has in-degree 0 (the root)
- All other nodes have in-degree 1

```python
# Check if undirected graph is a tree
g = Graph(directed=False)
g.add_edge('A', 'B')
g.add_edge('B', 'C')
g.add_edge('C', 'D')

print(g.is_tree())  # True
print(f"Edges: {g.get_edge_count()}, Vertices: {len(g.get_vertices())}")
# Edges: 3, Vertices: 4  (satisfies m = n - 1)

# Adding an edge creates a cycle - no longer a tree
g.add_edge('D', 'A')
print(g.is_tree())  # False (has cycle)

# Check if directed graph is a tree
g_dir = Graph(directed=True)
g_dir.add_edge('Root', 'A')
g_dir.add_edge('Root', 'B')
g_dir.add_edge('A', 'C')
g_dir.add_edge('A', 'D')

print(g_dir.is_tree())  # True (rooted tree with unique paths from root)
```

### Adjacency Matrix (Adjazenzmatrix)

**Theory:** For graph G = (V, E) with V = {v₁, ..., vₙ}, the adjacency matrix A ∈ ℝⁿˣⁿ stores edges:
- aᵢⱼ = 1 (or weight) if edge from vᵢ to vⱼ exists
- aᵢⱼ = 0 if no edge from vᵢ to vⱼ exists
- Undirected graphs → symmetric matrix
- Directed graphs → generally not symmetric

```python
# Get adjacency matrix
matrix = g.get_adjacency_matrix()
print(matrix)
# [[1, 1, 1, 0],
#  [1, 0, 1, 1],
#  [1, 1, 0, 0],
#  [0, 1, 0, 0]]

# Create graph from adjacency matrix
matrix = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]
vertices = ['A', 'B', 'C']
g = UndirectedGraph.from_adjacency_matrix(matrix, vertices)

# For weighted graphs, matrix contains weights
weighted_matrix = [
    [0, 5, 10],
    [5, 0, 3],
    [10, 3, 0]
]
g = WeightedUndirectedGraph.from_adjacency_matrix(weighted_matrix, vertices)
```

### Adjacency List Export/Import

The `get_adjacency_list()` method allows you to export the graph structure and recreate it later:

```python
from pyhelper_jkluess.Complex.Graphs.graph import Graph

# Create and populate a graph
g1 = Graph(directed=False, weighted=True)
g1.add_edge('A', 'B', 10)
g1.add_edge('B', 'C', 5)
g1.add_edge('A', 'C', 8)

# Export adjacency list
adj_list = g1.get_adjacency_list()
print(adj_list)
# Output: {'A': [('B', 10), ('C', 8)], 
#          'B': [('A', 10), ('C', 5)], 
#          'C': [('A', 8), ('B', 5)]}

# Create a new graph from the adjacency list
g2 = Graph(directed=False, weighted=True, data=adj_list)

# g2 is now an exact copy of g1
assert g2.vertex_count() == g1.vertex_count()
assert g2.edge_count() == g1.edge_count()
```

**Format:**
- **Unweighted graphs**: `Dict[vertex, List[neighbor]]`
- **Weighted graphs**: `Dict[vertex, List[(neighbor, weight)]]`

### Example: Complete Analysis

```python
from pyhelper_jkluess.Complex.Graphs.undirected_graph import UndirectedGraph

g = UndirectedGraph()
g.add_edge("A", "B")
g.add_edge("B", "C")
g.add_edge("C", "A")
g.add_edge("D", "E")

# Paths
path = g.find_path("A", "C")
print(f"Path A→C: {path}")
print(f"Path length: {g.path_length(path)}")
print(f"Simple path? {g.is_simple_path(path)}")

# Cycles
print(f"Has cycles? {g.has_cycle()}")
print(f"All cycles: {g.find_cycles()}")

# Connectivity
print(f"Connected? {g.is_connected()}")
components = g.get_connected_components()
print(f"Components: {components}")  # [{'A', 'B', 'C'}, {'D', 'E'}]

# Adjacency matrix
matrix = g.get_adjacency_matrix()
print(f"Adjacency matrix:\n{matrix}")

# Detailed analysis
g.print_graph_analysis()
```

## Implementation Details

### Unified Architecture

All specialized graph classes inherit from the base `Graph` class:

```python
class Graph:
    """Base class supporting all 4 graph types via parameters"""
    def __init__(self, directed=False, weighted=False, data=None):
        self._directed = directed
        self._weighted = weighted
        # ~25 core methods implemented here

class UndirectedGraph(Graph):
    """Convenience class for undirected unweighted graphs"""
    def __init__(self, data=None):
        super().__init__(directed=False, weighted=False, data=data)
    # Only ~6 specialized methods

class DirectedGraph(Graph):
    """Convenience class for directed unweighted graphs"""
    def __init__(self, data=None):
        super().__init__(directed=True, weighted=False, data=data)
    # Only ~6 specialized methods

class WeightedUndirectedGraph(Graph):
    """Convenience class for weighted undirected graphs"""
    def __init__(self, data=None):
        super().__init__(directed=False, weighted=True, data=data)
    # Only ~9 specialized methods

class WeightedDirectedGraph(Graph):
    """Convenience class for weighted directed graphs"""
    def __init__(self, data=None):
        super().__init__(directed=True, weighted=True, data=data)
    # Only ~8 specialized methods
```

### Import Compatibility

All classes support both import styles:

```python
# Package import (when using as module)
from pyhelper_jkluess.Complex.Graphs.graph import Graph
from pyhelper_jkluess.Complex.Graphs.undirected_graph import UndirectedGraph

# Direct import (when running scripts directly)
from graph import Graph
from undirected_graph import UndirectedGraph
```


