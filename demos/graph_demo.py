"""
Graph Data Structure - Comprehensive Demo

This demo showcases the complete functionality of the Graph class,
including all 4 graph types and their operations.

Topics covered:
1. Undirected unweighted graphs
2. Directed unweighted graphs
3. Undirected weighted graphs
4. Directed weighted graphs
5. Graph traversals (DFS, BFS)
6. Generator-based iteration
7. Shortest path algorithms
8. Cycle detection
9. Connectivity analysis
10. Minimum spanning trees
11. Import/export operations

Author: PyHelper JKluess
Date: December 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhelper_jkluess.Complex.Graphs.graph import Graph


def example_1_undirected_unweighted():
    """
    Example 1: Undirected unweighted graph
    
    Demonstrates:
    - Social network (friendships are bidirectional)
    - Basic graph operations
    - Connectivity
    """
    print("=" * 80)
    print("EXAMPLE 1: Undirected Unweighted Graph (Social Network)")
    print("=" * 80)
    
    # Create graph
    graph = Graph(directed=False, weighted=False)
    print("✓ Created undirected unweighted graph")
    
    # Add friendships
    friendships = [
        ('Alice', 'Bob'),
        ('Bob', 'Charlie'),
        ('Charlie', 'David'),
        ('David', 'Alice'),
        ('Bob', 'David')
    ]
    
    print(f"\n👥 Adding friendships:")
    for person1, person2 in friendships:
        graph.add_edge(person1, person2)
        print(f"  {person1} ←→ {person2}")
    
    print(f"\n📊 Graph Properties:")
    print(f"  Vertices (people): {graph.vertex_count()}")
    print(f"  Edges (friendships): {graph.edge_count()}")
    print(f"  Is connected: {graph.is_connected()}")
    
    print(f"\n🔍 Alice's friends: {graph.get_neighbors('Alice')}")
    print(f"  Alice's degree: {graph.degree('Alice')}")
    
    print(f"\n🛤️  Path from Alice to Charlie:")
    path = graph.find_path('Alice', 'Charlie')
    print(f"  {' → '.join(path)}")
    
    return graph


def example_2_directed_unweighted():
    """
    Example 2: Directed unweighted graph
    
    Demonstrates:
    - Task dependencies
    - Directed relationships
    - In-degree and out-degree
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Directed Unweighted Graph (Task Dependencies)")
    print("=" * 80)
    
    # Create directed graph
    graph = Graph(directed=True, weighted=False)
    print("✓ Created directed unweighted graph")
    
    # Add task dependencies (A → B means "A must be done before B")
    dependencies = [
        ('Design', 'Implement'),
        ('Implement', 'Test'),
        ('Test', 'Deploy'),
        ('Design', 'Documentation'),
        ('Documentation', 'Deploy')
    ]
    
    print(f"\n📋 Adding task dependencies:")
    for task1, task2 in dependencies:
        graph.add_edge(task1, task2)
        print(f"  {task1} → {task2}")
    
    print(f"\n📊 Graph Properties:")
    print(f"  Vertices (tasks): {graph.vertex_count()}")
    print(f"  Edges (dependencies): {graph.edge_count()}")
    
    print(f"\n🔍 Task 'Implement':")
    print(f"  Prerequisites: {graph.get_predecessors('Implement')}")
    print(f"  Next tasks: {graph.get_neighbors('Implement')}")
    print(f"  In-degree: {graph.in_degree('Implement')}")
    print(f"  Out-degree: {graph.out_degree('Implement')}")
    
    print(f"\n🔄 Has cycles: {graph.has_cycle()}")
    print(f"  (Cycles would indicate circular dependencies - bad!)")
    
    return graph


def example_3_weighted_undirected():
    """
    Example 3: Weighted undirected graph
    
    Demonstrates:
    - City road network
    - Distances between cities
    - Shortest path
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Weighted Undirected Graph (City Roads)")
    print("=" * 80)
    
    # Create weighted undirected graph
    graph = Graph(directed=False, weighted=True)
    print("✓ Created weighted undirected graph")
    
    # Add roads with distances (km)
    roads = [
        ('Berlin', 'Hamburg', 289),
        ('Berlin', 'Munich', 584),
        ('Hamburg', 'Munich', 776),
        ('Munich', 'Frankfurt', 392),
        ('Berlin', 'Frankfurt', 545),
        ('Hamburg', 'Frankfurt', 487)
    ]
    
    print(f"\n🛣️  Adding roads with distances:")
    for city1, city2, distance in roads:
        graph.add_edge(city1, city2, distance)
        print(f"  {city1} ←→ {city2}: {distance} km")
    
    print(f"\n📊 Graph Properties:")
    print(f"  Cities: {graph.vertex_count()}")
    print(f"  Roads: {graph.edge_count()}")
    print(f"  Total road length: {graph.total_weight()} km")
    
    print(f"\n🔍 Berlin connections:")
    neighbors = graph.get_weighted_neighbors('Berlin')
    for city, distance in neighbors:
        print(f"  → {city}: {distance} km")
    
    print(f"\n🛤️  Shortest path Berlin → Munich:")
    path, distance = graph.find_shortest_path('Berlin', 'Munich')
    print(f"  Route: {' → '.join(path)}")
    print(f"  Distance: {distance} km")
    
    return graph


def example_4_weighted_directed():
    """
    Example 4: Weighted directed graph
    
    Demonstrates:
    - Flight network with prices
    - One-way connections
    - Weighted degrees
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Weighted Directed Graph (Flight Network)")
    print("=" * 80)
    
    # Create weighted directed graph
    graph = Graph(directed=True, weighted=True)
    print("✓ Created weighted directed graph")
    
    # Add flights with prices (€)
    flights = [
        ('London', 'Paris', 120),
        ('Paris', 'Rome', 150),
        ('Rome', 'London', 200),
        ('London', 'Rome', 180),
        ('Paris', 'London', 100)
    ]
    
    print(f"\n✈️  Adding flights with prices:")
    for origin, dest, price in flights:
        graph.add_edge(origin, dest, price)
        print(f"  {origin} → {dest}: €{price}")
    
    print(f"\n📊 Graph Properties:")
    print(f"  Cities: {graph.vertex_count()}")
    print(f"  Flights: {graph.edge_count()}")
    
    print(f"\n🔍 London airport:")
    print(f"  Outbound flights: {graph.get_neighbors('London')}")
    print(f"  Inbound flights from: {graph.get_predecessors('London')}")
    print(f"  Total outbound cost: €{graph.weighted_out_degree('London')}")
    print(f"  Total inbound cost: €{graph.weighted_in_degree('London')}")
    
    print(f"\n🛤️  Cheapest route London → Rome:")
    path, cost = graph.find_shortest_path('London', 'Rome')
    print(f"  Route: {' → '.join(path)}")
    print(f"  Cost: €{cost}")
    
    return graph


def example_5_traversals(graph):
    """
    Example 5: Graph traversals
    
    Demonstrates:
    - Depth-First Search (DFS)
    - Breadth-First Search (BFS)
    - Different exploration patterns
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Graph Traversals")
    print("=" * 80)
    
    print(f"\n🔄 Depth-First Search (DFS) from first vertex:")
    vertices = list(graph.get_vertices())
    start = vertices[0]
    dfs_result = graph.dfs(start)
    print(f"  Order: {' → '.join(dfs_result)}")
    print("  (Explores as deep as possible before backtracking)")
    
    print(f"\n🔄 Breadth-First Search (BFS) from first vertex:")
    bfs_result = graph.bfs(start)
    print(f"  Order: {' → '.join(bfs_result)}")
    print("  (Explores level by level)")
    
    print(f"\n💡 DFS vs BFS:")
    print("  • DFS: Good for exploring all paths, detecting cycles")
    print("  • BFS: Good for shortest path (unweighted), level-order")


def example_6_generators(graph):
    """
    Example 6: Generator-based iteration
    
    Demonstrates:
    - Memory-efficient traversal
    - Early stopping
    - Processing during iteration
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Generator-based Traversal")
    print("=" * 80)
    
    vertices = list(graph.get_vertices())
    start = vertices[0] if vertices else None
    
    if start:
        print(f"\n🔄 DFS Generator (first 3 vertices):")
        for i, vertex in enumerate(graph.iter_dfs(start)):
            print(f"  {i+1}. {vertex}")
            if i >= 2:
                print("  ... (stopped early)")
                break
        
        print(f"\n🔄 BFS Generator with processing:")
        for vertex in graph.iter_bfs(start):
            print(f"  Processing: {vertex}")
            # Could do expensive operation here
            if len(vertex) > 10:  # Example condition
                print(f"  Found long name: {vertex}, stopping")
                break
        
        print(f"\n💡 Generators:")
        print("  • Yield one vertex at a time")
        print("  • Memory efficient for large graphs")
        print("  • Allow early exit with break")


def example_7_shortest_paths():
    """
    Example 7: Shortest path algorithms
    
    Demonstrates:
    - BFS for unweighted graphs
    - Dijkstra for weighted graphs
    - Path reconstruction
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Shortest Path Algorithms")
    print("=" * 80)
    
    # Unweighted graph
    print("\n📋 Unweighted Graph (BFS):")
    g1 = Graph(directed=False, weighted=False)
    for edge in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D')]:
        g1.add_edge(*edge)
    
    path = g1.find_path('A', 'D')
    print(f"  A → D: {' → '.join(path)}")
    print(f"  Length: {g1.path_length(path)} edges")
    
    # Weighted graph
    print("\n📋 Weighted Graph (Dijkstra):")
    g2 = Graph(directed=False, weighted=True)
    g2.add_edge('A', 'B', 4)
    g2.add_edge('B', 'C', 2)
    g2.add_edge('C', 'D', 3)
    g2.add_edge('A', 'D', 10)
    
    path, weight = g2.find_shortest_path('A', 'D')
    print(f"  A → D: {' → '.join(path)}")
    print(f"  Weight: {weight}")
    
    # Dijkstra's algorithm (all distances)
    distances = g2.dijkstra('A')
    print(f"\n🗺️  All distances from A:")
    for vertex, (dist, path) in distances.items():
        print(f"  → {vertex}: distance={dist}, path={' → '.join(path)}")


def example_8_cycles():
    """
    Example 8: Cycle detection
    
    Demonstrates:
    - Detecting cycles
    - Finding all cycles
    - Acyclic verification
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Cycle Detection")
    print("=" * 80)
    
    # Graph with cycle
    print("\n🔄 Graph with Cycle:")
    g1 = Graph(directed=False, weighted=False)
    for edge in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('B', 'D')]:
        g1.add_edge(*edge)
    
    print(f"  Has cycle: {g1.has_cycle()}")
    print(f"  Is acyclic: {g1.is_acyclic()}")
    
    cycles = g1.find_cycles()
    print(f"\n  Found {len(cycles)} cycle(s):")
    for i, cycle in enumerate(cycles[:3], 1):  # Show first 3
        print(f"    {i}. {' → '.join(cycle)}")
    
    # Acyclic graph (tree)
    print("\n🌳 Acyclic Graph (Tree):")
    g2 = Graph(directed=False, weighted=False)
    for edge in [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E')]:
        g2.add_edge(*edge)
    
    print(f"  Has cycle: {g2.has_cycle()}")
    print(f"  Is acyclic: {g2.is_acyclic()}")
    print(f"  Is tree: {g2.is_tree()}")


def example_9_connectivity():
    """
    Example 9: Connectivity analysis
    
    Demonstrates:
    - Connected components
    - Strong connectivity (directed)
    - Reachability
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Connectivity Analysis")
    print("=" * 80)
    
    # Disconnected graph
    print("\n📊 Disconnected Graph:")
    g1 = Graph(directed=False, weighted=False)
    g1.add_edge('A', 'B')
    g1.add_edge('B', 'C')
    g1.add_edge('D', 'E')
    g1.add_edge('F', 'G')
    
    print(f"  Is connected: {g1.is_connected()}")
    components = g1.get_connected_components()
    print(f"  Components: {len(components)}")
    for i, component in enumerate(components, 1):
        print(f"    {i}. {sorted(component)}")
    
    # Directed graph - strong connectivity
    print("\n📊 Directed Graph - Strong Connectivity:")
    g2 = Graph(directed=True, weighted=False)
    g2.add_edge('A', 'B')
    g2.add_edge('B', 'C')
    g2.add_edge('C', 'A')
    g2.add_edge('D', 'E')
    
    print(f"  Is strongly connected: {g2.is_strongly_connected()}")
    scc = g2.get_strongly_connected_components()
    print(f"  Strongly connected components:")
    for i, component in enumerate(scc, 1):
        print(f"    {i}. {sorted(component)}")
    
    # Reachability
    print(f"\n🛤️  Reachability:")
    print(f"  A can reach C: {g2.is_reachable('A', 'C')}")
    print(f"  C can reach A: {g2.is_reachable('C', 'A')}")
    print(f"  A can reach D: {g2.is_reachable('A', 'D')}")


def example_10_mst():
    """
    Example 10: Minimum Spanning Tree
    
    Demonstrates:
    - Kruskal's algorithm
    - Prim's algorithm
    - MST properties
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Minimum Spanning Tree (MST)")
    print("=" * 80)
    
    # Create weighted undirected graph
    graph = Graph(directed=False, weighted=True)
    edges = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'C', 1),
        ('B', 'D', 5),
        ('C', 'D', 8),
        ('C', 'E', 10),
        ('D', 'E', 2)
    ]
    
    print("\n🌐 Original Graph:")
    for v1, v2, w in edges:
        graph.add_edge(v1, v2, w)
        print(f"  {v1} ←→ {v2}: {w}")
    
    print(f"\n  Total weight: {graph.total_weight()}")
    
    # Kruskal's MST
    print("\n🌳 Minimum Spanning Tree (Kruskal):")
    mst_k = graph.minimum_spanning_tree_kruskal()
    if mst_k:
        kruskal_edges = mst_k.get_edges()
        total_weight = mst_k.total_weight()
        print(f"  Edges in MST:")
        for v1, v2, w in sorted(kruskal_edges, key=lambda x: x[2]):
            print(f"    {v1} ←→ {v2}: {w}")
        print(f"  MST total weight: {total_weight}")
    
    # Prim's MST
    print("\n🌳 Minimum Spanning Tree (Prim):")
    mst_p = graph.minimum_spanning_tree_prim()
    if mst_p:
        prim_edges = mst_p.get_edges()
        total_weight = mst_p.total_weight()
        print(f"  Edges in MST:")
        for v1, v2, w in sorted(prim_edges, key=lambda x: x[2]):
            print(f"    {v1} ←→ {v2}: {w}")
        print(f"  MST total weight: {total_weight}")
    
    print("\n💡 MST Properties:")
    print("  • Connects all vertices with minimum total weight")
    print("  • Has exactly n-1 edges for n vertices")
    print("  • No cycles")
    print("  • Both algorithms give same total weight (possibly different trees)")


def example_11_import_export():
    """
    Example 11: Import/Export operations
    
    Demonstrates:
    - Adjacency matrix conversion
    - Adjacency list conversion
    - Data preservation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 11: Import/Export Operations")
    print("=" * 80)
    
    # Create graph from adjacency list
    print("\n📋 Create from Adjacency List:")
    adj_list = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': []
    }
    
    graph = Graph(directed=False, weighted=False, data=adj_list)
    print(f"  Vertices: {sorted(graph.get_vertices())}")
    print(f"  Edges: {sorted(graph.get_edges())}")
    
    # Export to adjacency matrix
    print("\n📋 Export to Adjacency Matrix:")
    matrix = graph.get_adjacency_matrix()
    vertices = sorted(graph.get_vertices())
    print("     ", "  ".join(vertices))
    for i, row in enumerate(matrix):
        print(f"  {vertices[i]}  {row}")
    
    # Round-trip with weighted graph
    print("\n🔄 Round-trip with Weighted Graph:")
    g_weighted = Graph(directed=False, weighted=True)
    g_weighted.add_edge('X', 'Y', 10)
    g_weighted.add_edge('Y', 'Z', 20)
    
    # Export
    adj_list_export = g_weighted.get_adjacency_list()
    print(f"  Exported: {adj_list_export}")
    
    # Import
    g_reconstructed = Graph(directed=False, weighted=True, data=adj_list_export)
    print(f"  Reconstructed edges: {g_reconstructed.get_edges()}")
    print(f"  Weights preserved: {g_weighted.total_weight() == g_reconstructed.total_weight()}")


def example_12_advanced_operations():
    """
    Example 12: Advanced operations
    
    Demonstrates:
    - Degree sequences
    - Graph properties
    - Simple path checking
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 12: Advanced Operations")
    print("=" * 80)
    
    # Create sample graph
    graph = Graph(directed=False, weighted=False)
    for edge in [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'D')]:
        graph.add_edge(*edge)
    
    print("\n📊 Graph Properties:")
    print(f"  Vertices: {graph.vertex_count()}")
    print(f"  Edges: {graph.edge_count()}")
    print(f"  Is simple: {graph.is_simple_graph()}")
    
    print(f"\n📊 Degree Information:")
    for vertex in sorted(graph.get_vertices()):
        print(f"  {vertex}: degree = {graph.degree(vertex)}")
    
    # Check specific path
    print(f"\n🛤️  Path Analysis:")
    path = ['A', 'B', 'D']
    print(f"  Path: {' → '.join(path)}")
    print(f"  Is simple path: {graph.is_simple_path(path)}")
    print(f"  Path length: {graph.path_length(path)}")


def main():
    """Run all examples in sequence"""
    import sys
    import io
    # Fix Windows console encoding for emoji support
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n")
    print("🌐" * 40)
    print("GRAPH DATA STRUCTURE - COMPREHENSIVE DEMO")
    print("🌐" * 40)
    
    # Run all examples
    graph1 = example_1_undirected_unweighted()
    graph2 = example_2_directed_unweighted()
    graph3 = example_3_weighted_undirected()
    graph4 = example_4_weighted_directed()
    
    example_5_traversals(graph1)
    example_6_generators(graph1)
    example_7_shortest_paths()
    example_8_cycles()
    example_9_connectivity()
    example_10_mst()
    example_11_import_export()
    example_12_advanced_operations()
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)
    print("\n📚 Key Takeaways:")
    print("  1. Graphs model relationships between entities (vertices and edges)")
    print("  2. Four types: directed/undirected × weighted/unweighted")
    print("  3. DFS explores deep, BFS explores wide")
    print("  4. Generators allow memory-efficient iteration")
    print("  5. Shortest path: BFS (unweighted) or Dijkstra (weighted)")
    print("  6. Cycles can be detected and found")
    print("  7. Connected components show graph structure")
    print("  8. MST finds minimum-weight connected subgraph")
    print("  9. Graphs support import/export via matrix or list")
    print(" 10. Rich set of graph theory operations available")
    print("\n")


if __name__ == "__main__":
    main()
