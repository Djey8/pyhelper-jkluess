import networkx as nx
from typing import List, Set, Dict, Optional, Any

"""
Undirected Graph Implementation

This module provides a basic undirected graph data structure with common operations
like adding/removing vertices and edges, and visualization capabilities.
"""

import matplotlib.pyplot as plt


class UndirectedGraph:
    """
    A basic undirected graph implementation using adjacency list representation.
    """
    
    def __init__(self, data: Optional[Dict[Any, List[Any]]] = None):
        """
        Initialize an undirected graph.
        
        Args:
            data: Optional dictionary where keys are vertices and values are lists of adjacent vertices
        """
        self._adjacency_list: Dict[Any, Set[Any]] = {}
        
        if data:
            for vertex, neighbors in data.items():
                self.add_vertex(vertex)
                for neighbor in neighbors:
                    self.add_edge(vertex, neighbor)
    
    def add_vertex(self, vertex: Any) -> bool:
        """
        Add a vertex to the graph.
        
        Args:
            vertex: The vertex to add
            
        Returns:
            bool: True if vertex was added, False if it already exists
        """
        if vertex not in self._adjacency_list:
            self._adjacency_list[vertex] = set()
            return True
        return False
    
    def remove_vertex(self, vertex: Any) -> bool:
        """
        Remove a vertex and all its edges from the graph.
        
        Args:
            vertex: The vertex to remove
            
        Returns:
            bool: True if vertex was removed, False if it doesn't exist
        """
        if vertex not in self._adjacency_list:
            return False
        
        # Remove all edges to this vertex
        for neighbor in list(self._adjacency_list[vertex]):
            self._adjacency_list[neighbor].discard(vertex)
        
        # Remove the vertex itself
        del self._adjacency_list[vertex]
        return True
    
    def add_edge(self, vertex1: Any, vertex2: Any) -> bool:
        """
        Add an edge between two vertices.
        
        Args:
            vertex1: First vertex
            vertex2: Second vertex
            
        Returns:
            bool: True if edge was added, False if vertices don't exist or edge already exists
        """
        if vertex1 not in self._adjacency_list:
            self.add_vertex(vertex1)
        if vertex2 not in self._adjacency_list:
            self.add_vertex(vertex2)
        
        if vertex2 not in self._adjacency_list[vertex1]:
            self._adjacency_list[vertex1].add(vertex2)
            self._adjacency_list[vertex2].add(vertex1)
            return True
        return False
    
    def remove_edge(self, vertex1: Any, vertex2: Any) -> bool:
        """
        Remove an edge between two vertices.
        
        Args:
            vertex1: First vertex
            vertex2: Second vertex
            
        Returns:
            bool: True if edge was removed, False if edge doesn't exist
        """
        if (vertex1 in self._adjacency_list and 
            vertex2 in self._adjacency_list and
            vertex2 in self._adjacency_list[vertex1]):
            
            self._adjacency_list[vertex1].discard(vertex2)
            self._adjacency_list[vertex2].discard(vertex1)
            return True
        return False
    
    def has_vertex(self, vertex: Any) -> bool:
        """Check if a vertex exists in the graph."""
        return vertex in self._adjacency_list
    
    def has_edge(self, vertex1: Any, vertex2: Any) -> bool:
        """Check if an edge exists between two vertices."""
        return (vertex1 in self._adjacency_list and 
                vertex2 in self._adjacency_list[vertex1])
    
    def get_vertices(self) -> List[Any]:
        """Get all vertices in the graph."""
        return list(self._adjacency_list.keys())
    
    def get_neighbors(self, vertex: Any) -> List[Any]:
        """Get all neighbors of a vertex."""
        if vertex in self._adjacency_list:
            return list(self._adjacency_list[vertex])
        return []
    
    def get_edges(self) -> List[tuple]:
        """Get all edges in the graph."""
        edges = []
        visited = set()
        
        for vertex in self._adjacency_list:
            for neighbor in self._adjacency_list[vertex]:
                edge = tuple(sorted([vertex, neighbor]))
                if edge not in visited:
                    edges.append(edge)
                    visited.add(edge)
        
        return edges
    
    def vertex_count(self) -> int:
        """Get the number of vertices in the graph."""
        return len(self._adjacency_list)
    
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return len(self.get_edges())
    
    def visualize(self, title: str = "Undirected Graph", figsize: tuple = (10, 8), positions: Optional[Dict[Any, tuple]] = None):
        """
        Visualize the graph using matplotlib and networkx.
        
        Args:
            title: Title for the graph visualization
            figsize: Figure size as (width, height)
            positions: Optional dictionary mapping vertices to (x, y) coordinates
                      If None, uses spring layout for automatic positioning
        """
        if not self._adjacency_list:
            print("Graph is empty - nothing to visualize")
            return
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add vertices
        for vertex in self._adjacency_list:
            G.add_node(vertex)
        
        # Add edges
        for vertex1, vertex2 in self.get_edges():
            G.add_edge(vertex1, vertex2)
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Use custom positions if provided, otherwise use spring layout
        if positions:
            pos = positions
        else:
            pos = nx.spring_layout(G, seed=42)
        
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='lightblue',
                node_size=500,
                font_size=12,
                font_weight='bold',
                edge_color='gray',
                width=2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Path and reachability methods
    def find_path(self, start: Any, end: Any) -> Optional[List[Any]]:
        """
        Find a path from start vertex to end vertex using BFS.
        
        Theory: A path (Weg/Pfad) from v to v' is a sequence of vertices v₀, v₁, ..., vₖ where:
        - v₀ = v (start vertex)
        - vₖ = v' (end vertex)
        - {vᵢ, vᵢ₊₁} ∈ E for i = 0, ..., k-1
        - k is the length of the path
        
        Args:
            start: Starting vertex
            end: Target vertex
            
        Returns:
            List of vertices forming the path, or None if no path exists
        """
        if start not in self._adjacency_list or end not in self._adjacency_list:
            return None
        
        if start == end:
            return [start]
        
        # BFS to find shortest path
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            vertex, path = queue.pop(0)
            
            for neighbor in self._adjacency_list[vertex]:
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def is_reachable(self, start: Any, end: Any) -> bool:
        """
        Check if end vertex is reachable from start vertex.
        
        Theory: Vertex v is reachable from vertex u if there exists a path from u to v.
        
        Args:
            start: Starting vertex
            end: Target vertex
            
        Returns:
            True if end is reachable from start, False otherwise
        """
        return self.find_path(start, end) is not None
    
    def path_length(self, path: List[Any]) -> int:
        """
        Calculate the length of a path.
        
        Theory: The length of a path is the number of edges in the path (k in v₀, v₁, ..., vₖ).
        
        Args:
            path: List of vertices forming a path
            
        Returns:
            Length of the path (number of edges)
        """
        if not path or len(path) < 2:
            return 0
        return len(path) - 1
    
    def is_simple_path(self, path: List[Any]) -> bool:
        """
        Check if a path is simple (all vertices are pairwise distinct).
        
        Theory: A path is simple if all vertices in the path are pairwise different.
        
        Args:
            path: List of vertices forming a path
            
        Returns:
            True if the path is simple, False otherwise
        """
        if not path:
            return True
        return len(path) == len(set(path))
    
    # Cycle detection methods
    def has_cycle(self) -> bool:
        """
        Check if the graph contains any cycle.
        
        Theory: A cycle (Zyklus) is a path v₀, v₁, ..., vₖ where v₀ = vₖ.
        A circle (Kreis) is a cycle where v₀, v₁, ..., vₖ₋₁ are pairwise distinct.
        Trivial cycles (length 1 or 2) are often not considered.
        A graph without cycles is called acyclic (azyklisch).
        
        Returns:
            True if the graph contains a cycle, False otherwise
        """
        if not self._adjacency_list:
            return False
        
        visited = set()
        
        def dfs_has_cycle(vertex: Any, parent: Optional[Any]) -> bool:
            visited.add(vertex)
            
            for neighbor in self._adjacency_list[vertex]:
                if neighbor not in visited:
                    if dfs_has_cycle(neighbor, vertex):
                        return True
                elif neighbor != parent:  # Found a back edge
                    return True
            
            return False
        
        # Check each connected component
        for vertex in self._adjacency_list:
            if vertex not in visited:
                if dfs_has_cycle(vertex, None):
                    return True
        
        return False
    
    def find_cycles(self) -> List[List[Any]]:
        """
        Find all simple cycles in the graph.
        
        Theory: Returns all simple cycles (Kreise) where vertices are pairwise distinct.
        
        Returns:
            List of cycles, where each cycle is a list of vertices
        """
        cycles = []
        visited = set()
        
        def dfs_find_cycles(vertex: Any, parent: Optional[Any], path: List[Any]):
            visited.add(vertex)
            path.append(vertex)
            
            for neighbor in self._adjacency_list[vertex]:
                if neighbor not in visited:
                    dfs_find_cycles(neighbor, vertex, path[:])
                elif neighbor != parent and neighbor in path:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    # Normalize cycle (smallest vertex first) to avoid duplicates
                    min_idx = cycle.index(min(cycle))
                    normalized = cycle[min_idx:] + cycle[:min_idx]
                    if normalized not in cycles:
                        cycles.append(normalized)
            
            path.pop()
        
        for vertex in self._adjacency_list:
            if vertex not in visited:
                dfs_find_cycles(vertex, None, [])
        
        return cycles
    
    def is_acyclic(self) -> bool:
        """
        Check if the graph is acyclic (contains no cycles).
        
        Theory: An acyclic graph contains no cycles. An undirected acyclic graph is a forest.
        
        Returns:
            True if the graph is acyclic, False otherwise
        """
        return not self.has_cycle()
    
    # Connectivity methods
    def is_connected(self) -> bool:
        """
        Check if the graph is connected.
        
        Theory: An undirected graph G is connected (zusammenhängend) if every vertex is
        reachable from every other vertex.
        
        Returns:
            True if the graph is connected, False otherwise
        """
        if not self._adjacency_list:
            return True
        
        # Start DFS from any vertex
        start_vertex = next(iter(self._adjacency_list))
        visited = set()
        stack = [start_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            stack.extend(self._adjacency_list[vertex] - visited)
        
        return len(visited) == len(self._adjacency_list)
    
    def get_connected_components(self) -> List[Set[Any]]:
        """
        Get all connected components of the graph.
        
        Theory: A connected component (Zusammenhangskomponente) is a maximal connected
        subgraph. It represents equivalence classes of vertices with respect to the
        "reachable from" relation.
        
        Returns:
            List of sets, where each set contains vertices in a connected component
        """
        components = []
        visited = set()
        
        for vertex in self._adjacency_list:
            if vertex not in visited:
                # BFS to find all vertices in this component
                component = set()
                queue = [vertex]
                
                while queue:
                    v = queue.pop(0)
                    if v in visited:
                        continue
                    visited.add(v)
                    component.add(v)
                    queue.extend(self._adjacency_list[v] - visited)
                
                components.append(component)
        
        return components
    
    # Adjacency matrix methods
    def get_adjacency_matrix(self) -> List[List[int]]:
        """
        Get the adjacency matrix representation of the graph.
        
        Theory: For graph G = (V, E) with V = {v₁, ..., vₙ}, the adjacency matrix
        A ∈ ℝⁿˣⁿ is defined as:
        - aᵢⱼ = 1 if edge from vertex vᵢ to vⱼ exists
        - aᵢⱼ = 0 if no edge from vertex vᵢ to vⱼ exists
        
        For undirected graphs, the matrix is symmetric.
        
        Returns:
            n×n matrix (list of lists) where matrix[i][j] = 1 if edge exists, 0 otherwise
        """
        if not self._adjacency_list:
            return []
        
        # Create ordered list of vertices
        vertices = sorted(self._adjacency_list.keys())
        n = len(vertices)
        vertex_to_index = {v: i for i, v in enumerate(vertices)}
        
        # Initialize matrix with zeros
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        # Fill matrix
        for vertex in vertices:
            i = vertex_to_index[vertex]
            for neighbor in self._adjacency_list[vertex]:
                j = vertex_to_index[neighbor]
                matrix[i][j] = 1
        
        return matrix
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: List[List[int]], vertices: Optional[List[Any]] = None) -> 'UndirectedGraph':
        """
        Create a graph from an adjacency matrix.
        
        Theory: Converts an adjacency matrix representation back to a graph structure.
        
        Args:
            matrix: n×n adjacency matrix where matrix[i][j] = 1 indicates an edge
            vertices: Optional list of vertex labels. If None, uses integers 0 to n-1
            
        Returns:
            New UndirectedGraph instance
        """
        if not matrix:
            return cls()
        
        n = len(matrix)
        if vertices is None:
            vertices = list(range(n))
        elif len(vertices) != n:
            raise ValueError("Number of vertices must match matrix dimensions")
        
        graph = cls()
        
        # Add all vertices
        for vertex in vertices:
            graph.add_vertex(vertex)
        
        # Add edges from matrix
        for i in range(n):
            for j in range(i, n):  # Only check upper triangle for undirected
                if matrix[i][j] == 1:
                    graph.add_edge(vertices[i], vertices[j])
        
        return graph
    
    def __str__(self) -> str:
        """String representation of the graph."""
        if not self._adjacency_list:
            return "Empty graph"
        
        result = "Undirected Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = sorted(list(self._adjacency_list[vertex]))
            result += f"  {vertex}: {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the graph."""
        return f"UndirectedGraph(vertices={self.vertex_count()}, edges={self.edge_count()})"

    def degree(self, vertex: Any) -> int:
        """
        Get the degree of a vertex (number of incident edges).
        
        For undirected graphs, the degree is the number of edges connected to the vertex.
        
        Args:
            vertex: The vertex to get the degree for
            
        Returns:
            int: The degree of the vertex, or 0 if vertex doesn't exist
            
        Theory:
            In an undirected graph G = (V, E), the degree of a vertex v is the number
            of edges incident to v, denoted as deg(v) = |{u ∈ V : {u,v} ∈ E}|
        """
        if vertex not in self._adjacency_list:
            return 0
        return len(self._adjacency_list[vertex])

    def get_degree_sequence(self) -> List[int]:
        """
        Get the degree sequence of the graph (degrees of all vertices sorted in descending order).
        
        Returns:
            List[int]: Sorted list of vertex degrees in descending order
        """
        degrees = [self.degree(vertex) for vertex in self._adjacency_list]
        return sorted(degrees, reverse=True)

    def is_simple_graph(self) -> bool:
        """
        Check if the graph is simple (schlicht).
        
        A simple graph has no self-loops (edges from a vertex to itself).
        Since this is an undirected graph implementation, we only check for self-loops.
        
        Returns:
            bool: True if the graph is simple, False otherwise
            
        Theory:
            A graph is called simple (schlicht) if it contains no edge whose endpoints
            (or start and end nodes) are identical (no self-loops).
        """
        for vertex in self._adjacency_list:
            if vertex in self._adjacency_list[vertex]:
                return False
        return True

    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the graph structure.
        
        Returns:
            Dict containing various graph properties and statistics
            
        Theory:
            For an undirected graph G = (V, E):
            - Vertices (V): Set of nodes in the graph
            - Edges (E): Set of connections between vertices
            - Adjacency: Two vertices u,v are adjacent if {u,v} ∈ E
            - Incidence: A vertex v and edge e are incident if v is an endpoint of e
            - Degree: For vertex v, deg(v) is the number of edges incident to v
            - Simple graph: Contains no self-loops
        """
        if not self._adjacency_list:
            return {
                "vertices": 0,
                "edges": 0,
                "is_simple": True,
                "degree_sequence": [],
                "min_degree": 0,
                "max_degree": 0,
                "average_degree": 0.0,
                "vertex_degrees": {}
            }
        
        degrees = {vertex: self.degree(vertex) for vertex in self._adjacency_list}
        degree_values = list(degrees.values())
        
        return {
            "vertices": self.vertex_count(),
            "edges": self.edge_count(),
            "is_simple": self.is_simple_graph(),
            "degree_sequence": self.get_degree_sequence(),
            "min_degree": min(degree_values) if degree_values else 0,
            "max_degree": max(degree_values) if degree_values else 0,
            "average_degree": sum(degree_values) / len(degree_values) if degree_values else 0.0,
            "vertex_degrees": degrees
        }

    def print_graph_analysis(self):
        """
        Print a detailed analysis of the graph based on graph theory concepts.
        
        Theory explanation:
        - Graph G = (V, E) where V is the set of vertices and E is the set of edges
        - Adjacency: Vertices u,v are adjacent if there exists an edge {u,v} ∈ E
        - Incidence: A vertex v and edge e are incident if v is an endpoint of e
        - Degree: For undirected graphs, deg(v) = number of edges incident to vertex v
        - Simple graph: No self-loops (edges from a vertex to itself)
        """
        info = self.get_graph_info()
        
        print("=== Graph Theory Analysis ===")
        print(f"Graph G = (V, E) with |V| = {info['vertices']} vertices and |E| = {info['edges']} edges")
        print()
        
        print("Basic Properties:")
        print(f"  • Simple graph (schlicht): {'Yes' if info['is_simple'] else 'No'}")
        print(f"  • Minimum degree: {info['min_degree']}")
        print(f"  • Maximum degree: {info['max_degree']}")
        print(f"  • Average degree: {info['average_degree']:.2f}")
        print()
        
        print("Degree Information:")
        print(f"  • Degree sequence: {info['degree_sequence']}")
        print("  • Individual vertex degrees:")
        for vertex, degree in sorted(info['vertex_degrees'].items()):
            neighbors = sorted(self.get_neighbors(vertex))
            print(f"    deg({vertex}) = {degree}, adjacent to: {neighbors}")
        print()
        
        print("Graph Theory Concepts:")
        print("  • Adjacency: Two vertices u,v are adjacent if edge {u,v} exists")
        print("  • Incidence: A vertex and an edge are incident if the vertex is an endpoint")
        print("  • Degree: deg(v) = number of edges incident to vertex v")
        print("  • Simple: No self-loops (edges from vertex to itself)")

    
def main():
    """Test the UndirectedGraph implementation."""
    print("=== Testing UndirectedGraph ===\n")
    
    # Test 1: Empty graph
    print("1. Creating empty graph:")
    graph = UndirectedGraph()
    print(f"   {graph}")
    print(f"   Vertices: {graph.vertex_count()}, Edges: {graph.edge_count()}\n")
    
    # Test 2: Adding vertices and edges
    print("2. Adding vertices and edges:")
    graph.add_vertex('A')
    graph.add_vertex('B')
    graph.add_vertex('C')
    graph.add_edge('A', 'B')
    graph.add_edge('B', 'C')
    graph.add_edge('A', 'C')
    print(f"   {graph}")
    print(f"   Edges: {graph.get_edges()}\n")
    
    # Test 3: Graph with initial data
    print("3. Creating graph with initial data:")
    data = {
        'A': ['B', 'C', 'D'],       # Bottom left corner - connected to bottom right and top left
        'B': ['A', 'D'],       # Bottom right corner - connected to bottom left and top right  
        'C': ['A', 'D', 'E'],  # Top left corner - connected to bottom left, top right, and roof peak
        'D': ['B', 'C', 'E'],  # Top right corner - connected to bottom right, top left, and roof peak
        'E': ['C', 'D'],       # Roof peak - connected to both top corners
        'L': []
    }
    graph2 = UndirectedGraph(data)
    print(f"   {graph2}")
    print(f"   Vertices: {graph2.get_vertices()}")
    print(f"   Neighbors of C: {graph2.get_neighbors('C')}\n")
    
    # Test 4: Testing graph theory functionalities
    print("4. Testing degree and graph theory features:")
    print(f"   Degree of vertex A: {graph2.degree('A')}")
    print(f"   Degree of vertex E: {graph2.degree('E')}")
    print(f"   Degree sequence: {graph2.get_degree_sequence()}")
    print(f"   Is simple graph: {graph2.is_simple_graph()}\n")
    
    # Test 5: Comprehensive graph analysis
    print("5. Comprehensive graph analysis:")
    graph2.print_graph_analysis()
    print()
    
    # Test 6: Testing with self-loop (non-simple graph)
    print("6. Testing with self-loop:")
    graph3 = UndirectedGraph({'X': ['Y', 'X'], 'Y': ['X']})  # X has self-loop
    print(f"   Graph with self-loop: {graph3}")
    print(f"   Is simple: {graph3.is_simple_graph()}")
    print(f"   Degree of X (with self-loop): {graph3.degree('X')}\n")
    
    # Test 7: Testing removal operations
    print("7. Testing removal operations:")
    print(f"   Before removal: {graph2}")
    graph2.remove_edge('A', 'D')
    print(f"   After removing edge (A,D): {graph2}")
    graph2.remove_vertex('L')
    print(f"   After removing vertex L: {graph2}\n")
    
    # Test 8: Edge and vertex checking
    print("8. Testing existence checks:")
    print(f"   Has vertex A: {graph2.has_vertex('A')}")
    print(f"   Has vertex E: {graph2.has_vertex('E')}")
    print(f"   Has edge (A,C): {graph2.has_edge('A', 'C')}")
    print(f"   Has edge (A,B): {graph2.has_edge('A', 'B')}\n")
    
    # Test 9: Graph information summary
    print("9. Graph information summary:")
    info = graph2.get_graph_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    print()
    
    # Test 10: Visualization (comment out if matplotlib/networkx not available)
    print("10. Visualizing graph:")
    try:
        # Define positions for house-like structure
        house_positions = {
            'A': (0, 0),    # Bottom left corner
            'B': (2, 0),    # Bottom right corner  
            'C': (0, 2),    # Top left corner
            'D': (2, 2),    # Top right corner
            'E': (1, 3)     # Roof peak
        }
        graph2.visualize("Modified House Graph", positions=house_positions)
    except ImportError:
        print("   Matplotlib or NetworkX not available for visualization")
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()