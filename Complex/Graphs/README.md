# Graphs

Graph implementations with visualization support.

**Required:** `pip install networkx matplotlib`

## UndirectedGraph - Symmetric Edges

Edges work both ways: A—B means A connects to B and B connects to A.

```python
from Complex.Graphs.undirected_graph import UndirectedGraph

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

## DirectedGraph - One-Way Edges

Edges have direction: A→B doesn't mean B→A.

```python
from Complex.Graphs.directed_graph import DirectedGraph

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

## WeightedUndirectedGraph - Edges with Costs

Like UndirectedGraph but edges have weights (distances, costs, etc.).

```python
from Complex.Graphs.weighted_undirected_graph import WeightedUndirectedGraph

g = WeightedUndirectedGraph()
g.add_edge("Berlin", "Munich", 584)      # 584 km
g.add_edge("Munich", "Vienna", 434)      # 434 km
g.add_edge("Berlin", "Vienna", 680)      # 680 km

print(g.get_weight("Berlin", "Munich"))  # 584
print(g.get_neighbors_with_weights("Munich"))  # {'Berlin': 584, 'Vienna': 434}

# Find shortest path
path, distance = g.dijkstra("Berlin", "Vienna")
print(f"Best route: {path}, Distance: {distance} km")

g.visualize(title="City Network", show_weights=True)
```

**Use for:** Road networks, flight routes, weighted social connections.

## WeightedDirectedGraph - One-Way Edges with Costs

Like DirectedGraph but edges have weights.

```python
from Complex.Graphs.weighted_directed_graph import WeightedDirectedGraph

g = WeightedDirectedGraph()
g.add_edge("A", "B", 10)  # A → B costs 10
g.add_edge("B", "C", 5)   # B → C costs 5
g.add_edge("A", "C", 20)  # A → C costs 20

print(g.get_weight("A", "B"))  # 10
print(g.out_degree("A"))       # 2 outgoing edges

# Find shortest path
path, cost = g.dijkstra("A", "C")
print(f"Cheapest path: {path}, Cost: {cost}")  # A → B → C, cost 15

g.visualize(title="Task Network", show_weights=True)
```

**Use for:** Optimized task scheduling, network routing, weighted dependencies.

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
g.get_weight("A", "B")            # Get edge weight
g.get_neighbors_with_weights("A") # {neighbor: weight, ...}
g.dijkstra("A", "B")              # Shortest path (path, distance)

# DirectedGraph only
g.get_predecessors("A")           # Who points to A?
g.in_degree("A")                  # How many point to A?
g.out_degree("A")                 # How many does A point to?

# Graph theory (Directed & Undirected)
g.degree("A")                     # Total degree
g.is_simple_graph()               # Check for self-loops
g.get_graph_info()                # Statistics dict
g.print_graph_analysis()          # Detailed analysis

# Visualize
g.visualize(title="My Graph", figsize=(12, 8))
g.visualize(show_weights=True)    # Weighted graphs
```

## Graph Theory Concepts

All graph implementations support fundamental graph theory operations.

### Paths and Reachability (Wege und Erreichbarkeit)

**Theory:** A path (Weg/Pfad) from vertex `v` to `v'` is a sequence of vertices v₀, v₁, ..., vₖ where:
- v₀ = v (start vertex / Anfangsknoten)
- vₖ = v' (end vertex / Endknoten)
- Edges exist between consecutive vertices
- k is the length of the path (Länge des Wegs)

A path is **simple** if all vertices are pairwise distinct. Vertex v' is **reachable** from v if a path exists from v to v'.

```python
# Find a path between two vertices
path = g.find_path("A", "D")
print(path)  # ['A', 'B', 'C', 'D']

# Check if vertex is reachable
print(g.is_reachable("A", "D"))  # True

# Path properties
length = g.path_length(path)  # Number of edges
is_simple = g.is_simple_path(path)  # All vertices distinct?

# For weighted graphs: get path weight
weight = g.path_weight(path)  # Sum of edge weights
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
- **Connected** (zusammenhängend): Every vertex is reachable from every other vertex
- **Connected component** (Zusammenhangskomponente): A maximal connected subgraph
- Equivalence classes with respect to "reachable from" relation

**Directed Graphs:**
- **Strongly connected** (stark zusammenhängend): Every vertex is reachable from every other vertex following directed edges
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

### Example: Complete Analysis

```python
from Complex.Graphs.undirected_graph import UndirectedGraph

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
