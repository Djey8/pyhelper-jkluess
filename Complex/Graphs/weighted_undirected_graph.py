import networkx as nx
from typing import List, Set, Dict, Optional, Any, Union

import matplotlib.pyplot as plt


class WeightedUndirectedGraph:
    """
    A weighted undirected graph implementation using adjacency list representation.
    Each edge has an associated weight (numeric value).
    """
    
    def __init__(self, data: Optional[Dict[Any, Dict[Any, Union[int, float]]]] = None):
        """
        Initialize a weighted undirected graph.
        
        Args:
            data: Optional dictionary where keys are vertices and values are dictionaries
                  mapping adjacent vertices to their edge weights
        """
        self._adjacency_list: Dict[Any, Dict[Any, Union[int, float]]] = {}
        
        if data:
            for vertex, neighbors in data.items():
                self.add_vertex(vertex)
                for neighbor, weight in neighbors.items():
                    self.add_edge(vertex, neighbor, weight)
    
    def add_vertex(self, vertex: Any) -> bool:
        """
        Add a vertex to the graph.
        
        Args:
            vertex: The vertex to add
            
        Returns:
            bool: True if vertex was added, False if it already exists
        """
        if vertex not in self._adjacency_list:
            self._adjacency_list[vertex] = {}
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
        for neighbor in list(self._adjacency_list[vertex].keys()):
            del self._adjacency_list[neighbor][vertex]
        
        # Remove the vertex itself
        del self._adjacency_list[vertex]
        return True
    
    def add_edge(self, vertex1: Any, vertex2: Any, weight: Union[int, float] = 1) -> bool:
        """
        Add a weighted edge between two vertices.
        
        Args:
            vertex1: First vertex
            vertex2: Second vertex
            weight: Weight of the edge (default: 1)
            
        Returns:
            bool: True if edge was added, False if vertices don't exist or edge already exists
        """
        if vertex1 not in self._adjacency_list:
            self.add_vertex(vertex1)
        if vertex2 not in self._adjacency_list:
            self.add_vertex(vertex2)
        
        if vertex2 not in self._adjacency_list[vertex1]:
            self._adjacency_list[vertex1][vertex2] = weight
            self._adjacency_list[vertex2][vertex1] = weight
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
            
            del self._adjacency_list[vertex1][vertex2]
            del self._adjacency_list[vertex2][vertex1]
            return True
        return False
    
    def has_vertex(self, vertex: Any) -> bool:
        """Check if a vertex exists in the graph."""
        return vertex in self._adjacency_list
    
    def has_edge(self, vertex1: Any, vertex2: Any) -> bool:
        """Check if an edge exists between two vertices."""
        return (vertex1 in self._adjacency_list and 
                vertex2 in self._adjacency_list[vertex1])
    
    def get_edge_weight(self, vertex1: Any, vertex2: Any) -> Optional[Union[int, float]]:
        """
        Get the weight of an edge between two vertices.
        
        Args:
            vertex1: First vertex
            vertex2: Second vertex
            
        Returns:
            The weight of the edge, or None if edge doesn't exist
        """
        if self.has_edge(vertex1, vertex2):
            return self._adjacency_list[vertex1][vertex2]
        return None
    
    def update_edge_weight(self, vertex1: Any, vertex2: Any, new_weight: Union[int, float]) -> bool:
        """
        Update the weight of an existing edge.
        
        Args:
            vertex1: First vertex
            vertex2: Second vertex
            new_weight: New weight for the edge
            
        Returns:
            bool: True if weight was updated, False if edge doesn't exist
        """
        if self.has_edge(vertex1, vertex2):
            self._adjacency_list[vertex1][vertex2] = new_weight
            self._adjacency_list[vertex2][vertex1] = new_weight
            return True
        return False
    
    def get_vertices(self) -> List[Any]:
        """Get all vertices in the graph."""
        return list(self._adjacency_list.keys())
    
    def get_neighbors(self, vertex: Any) -> List[Any]:
        """Get all neighbors of a vertex."""
        if vertex in self._adjacency_list:
            return list(self._adjacency_list[vertex].keys())
        return []
    
    def get_weighted_neighbors(self, vertex: Any) -> Dict[Any, Union[int, float]]:
        """Get all neighbors of a vertex with their edge weights."""
        if vertex in self._adjacency_list:
            return dict(self._adjacency_list[vertex])
        return {}
    
    def get_edges(self) -> List[tuple]:
        """Get all edges in the graph as (vertex1, vertex2, weight) tuples."""
        edges = []
        visited = set()
        
        for vertex in self._adjacency_list:
            for neighbor, weight in self._adjacency_list[vertex].items():
                edge_key = tuple(sorted([vertex, neighbor]))
                if edge_key not in visited:
                    edges.append((vertex, neighbor, weight))
                    visited.add(edge_key)
        
        return edges
    
    def vertex_count(self) -> int:
        """Get the number of vertices in the graph."""
        return len(self._adjacency_list)
    
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return len(self.get_edges())
    
    def total_weight(self) -> Union[int, float]:
        """Get the total weight of all edges in the graph."""
        return sum(weight for _, _, weight in self.get_edges())
    
    def visualize(self, title: str = "Weighted Undirected Graph", figsize: tuple = (10, 8), 
                  positions: Optional[Dict[Any, tuple]] = None, show_weights: bool = True):
        """
        Visualize the weighted graph using matplotlib and networkx.
        
        Args:
            title: Title for the graph visualization
            figsize: Figure size as (width, height)
            positions: Optional dictionary mapping vertices to (x, y) coordinates
            show_weights: Whether to display edge weights on the graph
        """
        if not self._adjacency_list:
            print("Graph is empty - nothing to visualize")
            return
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add vertices
        for vertex in self._adjacency_list:
            G.add_node(vertex)
        
        # Add weighted edges
        for vertex1, vertex2, weight in self.get_edges():
            G.add_edge(vertex1, vertex2, weight=weight)
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Use custom positions if provided, otherwise use spring layout
        if positions:
            pos = positions
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes and edges
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='lightblue',
                node_size=500,
                font_size=12,
                font_weight='bold',
                edge_color='gray',
                width=2)
        
        # Draw edge labels (weights) if requested
        if show_weights:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Path and reachability methods
    def find_path(self, start: Any, end: Any) -> Optional[List[Any]]:
        """
        Find a path from start vertex to end vertex using BFS (unweighted shortest path).
        
        Theory: A path (Weg/Pfad) from v to v' is a sequence of vertices v₀, v₁, ..., vₖ where:
        - v₀ = v (start vertex)
        - vₖ = v' (end vertex)
        - {vᵢ, vᵢ₊₁} ∈ E for i = 0, ..., k-1
        - k is the length of the path
        
        Note: This finds a path ignoring weights. For weighted shortest path, use dijkstra().
        
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
        
        # BFS to find shortest path (by number of edges)
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
        Calculate the length of a path (number of edges).
        
        Theory: The length of a path is the number of edges in the path (k in v₀, v₁, ..., vₖ).
        
        Args:
            path: List of vertices forming a path
            
        Returns:
            Length of the path (number of edges)
        """
        if not path or len(path) < 2:
            return 0
        return len(path) - 1
    
    def path_weight(self, path: List[Any]) -> Union[int, float]:
        """
        Calculate the total weight of a path.
        
        Args:
            path: List of vertices forming a path
            
        Returns:
            Sum of edge weights in the path, or 0 if path is invalid
        """
        if not path or len(path) < 2:
            return 0
        
        total = 0
        for i in range(len(path) - 1):
            if path[i] not in self._adjacency_list or path[i+1] not in self._adjacency_list[path[i]]:
                return 0  # Invalid path
            total += self._adjacency_list[path[i]][path[i+1]]
        
        return total
    
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
            for neighbor in self._adjacency_list[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
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
                    for neighbor in self._adjacency_list[v]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    # Adjacency matrix methods
    def get_adjacency_matrix(self) -> List[List[Union[int, float]]]:
        """
        Get the weighted adjacency matrix representation of the graph.
        
        Theory: For weighted graph G = (V, E) with V = {v₁, ..., vₙ}, the weighted
        adjacency matrix A ∈ ℝⁿˣⁿ is defined as:
        - aᵢⱼ = weight if edge from vertex vᵢ to vⱼ exists
        - aᵢⱼ = 0 if no edge from vertex vᵢ to vⱼ exists
        
        For undirected graphs, the matrix is symmetric.
        
        Returns:
            n×n matrix where matrix[i][j] = edge weight if edge exists, 0 otherwise
        """
        if not self._adjacency_list:
            return []
        
        # Create ordered list of vertices
        vertices = sorted(self._adjacency_list.keys())
        n = len(vertices)
        vertex_to_index = {v: i for i, v in enumerate(vertices)}
        
        # Initialize matrix with zeros
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        # Fill matrix with weights
        for vertex in vertices:
            i = vertex_to_index[vertex]
            for neighbor, weight in self._adjacency_list[vertex].items():
                j = vertex_to_index[neighbor]
                matrix[i][j] = weight
        
        return matrix
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: List[List[Union[int, float]]], vertices: Optional[List[Any]] = None) -> 'WeightedUndirectedGraph':
        """
        Create a weighted graph from an adjacency matrix.
        
        Theory: Converts a weighted adjacency matrix representation back to a graph structure.
        
        Args:
            matrix: n×n adjacency matrix where matrix[i][j] contains the edge weight (0 = no edge)
            vertices: Optional list of vertex labels. If None, uses integers 0 to n-1
            
        Returns:
            New WeightedUndirectedGraph instance
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
                if matrix[i][j] != 0:
                    graph.add_edge(vertices[i], vertices[j], matrix[i][j])
        
        return graph
    
    def __str__(self) -> str:
        """String representation of the graph."""
        if not self._adjacency_list:
            return "Empty weighted graph"
        
        result = "Weighted Undirected Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = []
            for neighbor, weight in sorted(self._adjacency_list[vertex].items()):
                neighbors.append(f"{neighbor}({weight})")
            result += f"  {vertex}: {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the graph."""
        return f"WeightedUndirectedGraph(vertices={self.vertex_count()}, edges={self.edge_count()}, total_weight={self.total_weight()})"

    def degree(self, vertex: Any) -> int:
        """
        Get the degree of a vertex (number of incident edges).
        
        Args:
            vertex: The vertex to get the degree for
            
        Returns:
            int: The degree of the vertex, or 0 if vertex doesn't exist
        """
        if vertex not in self._adjacency_list:
            return 0
        return len(self._adjacency_list[vertex])
    
    def weighted_degree(self, vertex: Any) -> Union[int, float]:
        """
        Get the weighted degree of a vertex (sum of weights of incident edges).
        
        Args:
            vertex: The vertex to get the weighted degree for
            
        Returns:
            The sum of weights of all edges incident to the vertex, or 0 if vertex doesn't exist
        """
        if vertex not in self._adjacency_list:
            return 0
        return sum(self._adjacency_list[vertex].values())

    def get_degree_sequence(self) -> List[int]:
        """Get the degree sequence of the graph (degrees of all vertices sorted in descending order)."""
        degrees = [self.degree(vertex) for vertex in self._adjacency_list]
        return sorted(degrees, reverse=True)
    
    def get_weighted_degree_sequence(self) -> List[Union[int, float]]:
        """Get the weighted degree sequence of the graph sorted in descending order."""
        weighted_degrees = [self.weighted_degree(vertex) for vertex in self._adjacency_list]
        return sorted(weighted_degrees, reverse=True)

    def is_simple_graph(self) -> bool:
        """
        Check if the graph is simple (no self-loops).
        
        Returns:
            bool: True if the graph is simple, False otherwise
        """
        for vertex in self._adjacency_list:
            if vertex in self._adjacency_list[vertex]:
                return False
        return True
    
    def get_minimum_weight_edge(self) -> Optional[tuple]:
        """
        Get the edge with minimum weight.
        
        Returns:
            Tuple of (vertex1, vertex2, weight) for minimum weight edge, or None if no edges
        """
        edges = self.get_edges()
        if not edges:
            return None
        return min(edges, key=lambda x: x[2])
    
    def get_maximum_weight_edge(self) -> Optional[tuple]:
        """
        Get the edge with maximum weight.
        
        Returns:
            Tuple of (vertex1, vertex2, weight) for maximum weight edge, or None if no edges
        """
        edges = self.get_edges()
        if not edges:
            return None
        return max(edges, key=lambda x: x[2])

    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the weighted graph structure.
        
        Returns:
            Dict containing various graph properties and statistics
        """
        if not self._adjacency_list:
            return {
                "vertices": 0,
                "edges": 0,
                "total_weight": 0,
                "is_simple": True,
                "degree_sequence": [],
                "weighted_degree_sequence": [],
                "min_degree": 0,
                "max_degree": 0,
                "average_degree": 0.0,
                "min_weight": None,
                "max_weight": None,
                "average_weight": 0.0,
                "vertex_degrees": {},
                "vertex_weighted_degrees": {}
            }
        
        degrees = {vertex: self.degree(vertex) for vertex in self._adjacency_list}
        weighted_degrees = {vertex: self.weighted_degree(vertex) for vertex in self._adjacency_list}
        degree_values = list(degrees.values())
        edges = self.get_edges()
        weights = [weight for _, _, weight in edges] if edges else []
        
        return {
            "vertices": self.vertex_count(),
            "edges": self.edge_count(),
            "total_weight": self.total_weight(),
            "is_simple": self.is_simple_graph(),
            "degree_sequence": self.get_degree_sequence(),
            "weighted_degree_sequence": self.get_weighted_degree_sequence(),
            "min_degree": min(degree_values) if degree_values else 0,
            "max_degree": max(degree_values) if degree_values else 0,
            "average_degree": sum(degree_values) / len(degree_values) if degree_values else 0.0,
            "min_weight": min(weights) if weights else None,
            "max_weight": max(weights) if weights else None,
            "average_weight": sum(weights) / len(weights) if weights else 0.0,
            "vertex_degrees": degrees,
            "vertex_weighted_degrees": weighted_degrees
        }

    def print_graph_analysis(self):
        """
        Print a detailed analysis of the weighted graph based on graph theory concepts.
        """
        info = self.get_graph_info()
        
        print("=== Weighted Graph Theory Analysis ===")
        print(f"Weighted Graph G = (V, E, w) with |V| = {info['vertices']} vertices and |E| = {info['edges']} edges")
        print(f"Total weight of all edges: {info['total_weight']}")
        print()
        
        print("Basic Properties:")
        print(f"  • Simple graph (schlicht): {'Yes' if info['is_simple'] else 'No'}")
        print(f"  • Minimum degree: {info['min_degree']}")
        print(f"  • Maximum degree: {info['max_degree']}")
        print(f"  • Average degree: {info['average_degree']:.2f}")
        print()
        
        print("Weight Properties:")
        if info['min_weight'] is not None:
            print(f"  • Minimum edge weight: {info['min_weight']}")
            print(f"  • Maximum edge weight: {info['max_weight']}")
            print(f"  • Average edge weight: {info['average_weight']:.2f}")
            min_edge = self.get_minimum_weight_edge()
            max_edge = self.get_maximum_weight_edge()
            print(f"  • Minimum weight edge: {min_edge[0]}-{min_edge[1]} (weight: {min_edge[2]})")
            print(f"  • Maximum weight edge: {max_edge[0]}-{max_edge[1]} (weight: {max_edge[2]})")
        else:
            print("  • No edges in graph")
        print()
        
        print("Degree Information:")
        print(f"  • Degree sequence: {info['degree_sequence']}")
        print(f"  • Weighted degree sequence: {info['weighted_degree_sequence']}")
        print("  • Individual vertex information:")
        for vertex in sorted(self._adjacency_list.keys()):
            degree = info['vertex_degrees'][vertex]
            weighted_degree = info['vertex_weighted_degrees'][vertex]
            neighbors = self.get_weighted_neighbors(vertex)
            neighbor_str = ", ".join([f"{n}({w})" for n, w in sorted(neighbors.items())])
            print(f"    {vertex}: deg={degree}, weighted_deg={weighted_degree:.1f}, neighbors=[{neighbor_str}]")
        print()
        
        print("Weighted Graph Theory Concepts:")
        print("  • Weight function: w: E → ℝ assigns a weight to each edge")
        print("  • Weighted degree: sum of weights of all incident edges")
        print("  • Total weight: sum of all edge weights in the graph")


def main():
    """Test the WeightedUndirectedGraph implementation."""
    print("=== Testing WeightedUndirectedGraph ===\n")
    
    # Test 1: Empty graph
    print("1. Creating empty weighted graph:")
    graph = WeightedUndirectedGraph()
    print(f"   {graph}")
    print(f"   Vertices: {graph.vertex_count()}, Edges: {graph.edge_count()}, Total Weight: {graph.total_weight()}\n")
    
    # Test 2: Adding vertices and weighted edges
    print("2. Adding vertices and weighted edges:")
    graph.add_vertex('A')
    graph.add_vertex('B')
    graph.add_vertex('C')
    graph.add_edge('A', 'B', 5)
    graph.add_edge('B', 'C', 3)
    graph.add_edge('A', 'C', 7)
    print(f"   {graph}")
    print(f"   Edges: {graph.get_edges()}\n")
    
    # Test 3: Weighted graph with initial data
    print("3. Creating weighted graph with initial data:")
    data = {
        'A': {'B': 4, 'C': 2},       # A connected to B(weight 4) and C(weight 2)
        'B': {'A': 4, 'D': 3},       # B connected to A(4), D(3), E(6)
        'C': {'A': 2, 'D': 1, 'F': 8},  # C connected to A(2), D(1), F(8)
        'D': {'B': 3, 'C': 1, 'E': 2},  # D connected to B(3), C(1), E(2)
        'E': {'C': 6, 'D': 2},       # E connected to B(6), D(2)
        'F': {'C': 8},               # F connected to C(8)
        'G': {}                      # Isolated vertex
    }
    graph2 = WeightedUndirectedGraph(data)
    print(f"   {graph2}")
    print(f"   Total weight: {graph2.total_weight()}")
    print(f"   Weighted neighbors of B: {graph2.get_weighted_neighbors('B')}\n")
    
    # Test 4: Testing weighted graph theory functionalities
    print("4. Testing degree and weighted degree features:")
    print(f"   Degree of vertex B: {graph2.degree('B')}")
    print(f"   Weighted degree of vertex B: {graph2.weighted_degree('B')}")
    print(f"   Degree sequence: {graph2.get_degree_sequence()}")
    print(f"   Weighted degree sequence: {graph2.get_weighted_degree_sequence()}")
    print(f"   Is simple graph: {graph2.is_simple_graph()}\n")
    
    # Test 5: Weight-specific operations
    print("5. Testing weight-specific operations:")
    print(f"   Weight of edge (A,B): {graph2.get_edge_weight('A', 'B')}")
    print(f"   Minimum weight edge: {graph2.get_minimum_weight_edge()}")
    print(f"   Maximum weight edge: {graph2.get_maximum_weight_edge()}")
    
    # Update edge weight
    graph2.update_edge_weight('A', 'B', 10)
    print(f"   After updating (A,B) weight to 10: {graph2.get_edge_weight('A', 'B')}\n")
    
    # Test 6: Comprehensive weighted graph analysis
    print("6. Comprehensive weighted graph analysis:")
    graph2.print_graph_analysis()
    print()
    
    # Test 7: Testing with self-loop (non-simple graph)
    print("7. Testing with weighted self-loop:")
    graph3 = WeightedUndirectedGraph({'X': {'Y': 3, 'X': 5}, 'Y': {'X': 3}})
    print(f"   Graph with self-loop: {graph3}")
    print(f"   Is simple: {graph3.is_simple_graph()}")
    print(f"   Weighted degree of X (with self-loop): {graph3.weighted_degree('X')}\n")
    
    # Test 8: Testing removal operations
    print("8. Testing removal operations:")
    print(f"   Before removal: Total weight = {graph2.total_weight()}")
    graph2.remove_edge('F', 'C')  # Remove heaviest edge
    print(f"   After removing edge (F,C): Total weight = {graph2.total_weight()}")
    graph2.remove_vertex('G')  # Remove isolated vertex
    graph2.remove_vertex('F')
    print(f"   After removing isolated vertex G: {graph2.vertex_count()} vertices\n")
    
    # Test 9: Graph information summary
    print("9. Weighted graph information summary:")
    info = graph2.get_graph_info()
    for key, value in info.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    print()
    
    # Test 10: Visualization
    print("10. Visualizing weighted graph:")
    try:
        # Define positions for network-like structure
        house_positions = {
            'A': (0, 0),    # Bottom left corner
            'B': (2, 0),    # Bottom right corner  
            'C': (0, 2),    # Top left corner
            'D': (2, 2),    # Top right corner
            'E': (1, 3)     # Roof peak
        }
        graph2.visualize("Weighted Network Graph", positions=house_positions, show_weights=True)
    except ImportError:
        print("   Matplotlib or NetworkX not available for visualization")
    
    print("\n=== All weighted graph tests completed ===")


if __name__ == "__main__":
    main()