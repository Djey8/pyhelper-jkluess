from typing import List, Dict, Optional, Any, Union
try:
    from .graph import Graph
except ImportError:
    from graph import Graph

"""
Weighted Undirected Graph Implementation

This module provides a weighted undirected graph that inherits from the base Graph class.
All common functionality is inherited; this class only provides convenience methods
and specific behavior for weighted undirected graphs.
"""


class WeightedUndirectedGraph(Graph):
    """
    A weighted undirected graph implementation.
    Inherits all functionality from Graph with directed=False, weighted=True.
    """
    
    def __init__(self, data: Optional[Dict[Any, Dict[Any, Union[int, float]]]] = None):
        """
        Initialize a weighted undirected graph.
        
        Args:
            data: Optional dictionary where keys are vertices and values are dictionaries
                  mapping adjacent vertices to their edge weights
        """
        super().__init__(directed=False, weighted=True, data=data)
    
    # All core methods (add_vertex, remove_vertex, add_edge, remove_edge, has_vertex, 
    # has_edge, get_vertices, get_neighbors, get_edges, vertex_count, edge_count,
    # update_edge_weight, get_edge_weight) are inherited from 
    # Graph and work correctly for weighted undirected graphs.
    
    def get_weighted_neighbors(self, vertex: Any) -> Dict[Any, Union[int, float]]:
        """
        Get all neighbors of a vertex with their edge weights.
        
        Args:
            vertex: The vertex to get neighbors for
            
        Returns:
            Dictionary mapping neighbor vertices to their edge weights
        """
        if vertex in self._adjacency_list:
            return dict(self._adjacency_list[vertex])
        return {}
    
    # Degree methods (degree, weighted_degree) are inherited from Graph.
    
    # Path and reachability methods (find_path, is_reachable, is_simple_path, path_length,
    # path_weight) are inherited from Graph.
    
    # Cycle detection methods (has_cycle, find_cycles, is_acyclic) 
    # are inherited from Graph.
    
    # Connectivity methods (is_connected, get_connected_components) 
    # are inherited from Graph.
    
    # Adjacency matrix methods (get_adjacency_matrix) are inherited from Graph.
    
    # Visualization method (visualize) is inherited from Graph.
    
    # Weight method (total_weight) is inherited from Graph.
    
    def get_weight_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about edge weights in the graph.
        
        Returns:
            Dictionary with weight statistics
        """
        if not self.get_edges():
            return {'min_weight': 0, 'max_weight': 0, 'average_weight': 0, 'total_weight': 0}
        
        weights = [weight for _, _, weight in self.get_edges()]
        return {
            'min_weight': min(weights),
            'max_weight': max(weights),
            'average_weight': sum(weights) / len(weights),
            'total_weight': sum(weights)
        }
    
    def get_minimum_weight_edge(self) -> Optional[tuple]:
        """
        Get the edge with minimum weight.
        
        Returns:
            Tuple (vertex1, vertex2, weight) or None if no edges exist
        """
        edges = self.get_edges()
        if not edges:
            return None
        return min(edges, key=lambda e: e[2])
    
    def get_maximum_weight_edge(self) -> Optional[tuple]:
        """
        Get the edge with maximum weight.
        
        Returns:
            Tuple (vertex1, vertex2, weight) or None if no edges exist
        """
        edges = self.get_edges()
        if not edges:
            return None
        return max(edges, key=lambda e: e[2])
    
    def get_degree_sequence(self) -> List[int]:
        """
        Get the degree sequence of the graph.
        
        Returns:
            List of degrees sorted in descending order
        """
        degrees = [self.degree(vertex) for vertex in self._adjacency_list]
        return sorted(degrees, reverse=True)
    
    def get_weighted_degree_sequence(self) -> List[Union[int, float]]:
        """
        Get the weighted degree sequence of the graph.
        
        Returns:
            List of weighted degrees sorted in descending order
        """
        weighted_degrees = [self.weighted_degree(vertex) for vertex in self._adjacency_list]
        return sorted(weighted_degrees, reverse=True)
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the weighted undirected graph structure.
        
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
                "min_weighted_degree": 0,
                "max_weighted_degree": 0,
                "average_weighted_degree": 0.0,
                "is_connected": True,
                "connected_components": 0,
                "weight_statistics": {}
            }
        
        degrees = [self.degree(vertex) for vertex in self._adjacency_list]
        weighted_degrees = [self.weighted_degree(vertex) for vertex in self._adjacency_list]
        
        return {
            "vertices": self.vertex_count(),
            "edges": self.edge_count(),
            "total_weight": self.total_weight(),
            "is_simple": self.is_simple_graph(),
            "degree_sequence": self.get_degree_sequence(),
            "weighted_degree_sequence": self.get_weighted_degree_sequence(),
            "min_degree": min(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "average_degree": sum(degrees) / len(degrees) if degrees else 0.0,
            "min_weighted_degree": min(weighted_degrees) if weighted_degrees else 0,
            "max_weighted_degree": max(weighted_degrees) if weighted_degrees else 0,
            "average_weighted_degree": sum(weighted_degrees) / len(weighted_degrees) if weighted_degrees else 0.0,
            "is_connected": self.is_connected(),
            "connected_components": len(self.get_connected_components()),
            "weight_statistics": self.get_weight_statistics()
        }
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: List[List[Union[int, float]]], 
                            vertices: Optional[List[Any]] = None) -> 'WeightedUndirectedGraph':
        """
        Create a weighted undirected graph from an adjacency matrix.
        
        Args:
            matrix: n×n adjacency matrix where matrix[i][j] is the weight (0 means no edge)
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
        
        # Add edges with weights from matrix (only upper triangle for undirected)
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] != 0:
                    graph.add_edge(vertices[i], vertices[j], matrix[i][j])
        
        return graph
    
    def __str__(self) -> str:
        """String representation of the weighted undirected graph."""
        if not self._adjacency_list:
            return "Empty graph"
        
        result = "Weighted Undirected Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = sorted([(neighbor, weight) for neighbor, weight in self._adjacency_list[vertex].items()])
            result += f"  {vertex}: {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the weighted undirected graph."""
        return f"WeightedUndirectedGraph(vertices={self.vertex_count()}, edges={self.edge_count()}, total_weight={self.total_weight()})"

    def print_graph_analysis(self):
        """
        Print a detailed analysis of the weighted undirected graph based on graph theory concepts.
        """
        info = self.get_graph_info()
        
        print("=== Weighted Undirected Graph Theory Analysis ===")
        print(f"Weighted Undirected Graph G = (V, E, w) with |V| = {info['vertices']} vertices and |E| = {info['edges']} weighted edges")
        print(f"Total weight w(G) = {info['total_weight']}")
        print()
        
        print("Basic Properties:")
        print(f"  • Simple graph (schlicht): {'Yes' if info['is_simple'] else 'No'}")
        print(f"  • Minimum degree: {info['min_degree']}")
        print(f"  • Maximum degree: {info['max_degree']}")
        print(f"  • Average degree: {info['average_degree']:.2f}")
        print(f"  • Connected: {'Yes' if info['is_connected'] else 'No'}")
        print(f"  • Connected components: {info['connected_components']}")
        print()
        
        print("Weighted Degree Properties:")
        print(f"  • Minimum weighted degree: {info['min_weighted_degree']}")
        print(f"  • Maximum weighted degree: {info['max_weighted_degree']}")
        print(f"  • Average weighted degree: {info['average_weighted_degree']:.2f}")
        print()
        
        print("Degree Sequences:")
        print(f"  • Degree sequence: {info['degree_sequence']}")
        print(f"  • Weighted degree sequence: {info['weighted_degree_sequence']}")
        print()
        
        print("Weight Statistics:")
        stats = info['weight_statistics']
        if stats and stats.get('total_weight', 0) > 0:
            print(f"  • Minimum edge weight: {stats.get('min_weight', 'N/A')}")
            print(f"  • Maximum edge weight: {stats.get('max_weight', 'N/A')}")
            print(f"  • Average edge weight: {stats.get('average_weight', 0):.2f}")
        print()
        
        print("Individual Vertex Analysis:")
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = sorted(self.get_neighbors(vertex))
            weighted_neighbors = sorted(self.get_weighted_neighbors(vertex))
            print(f"  Vertex {vertex}:")
            print(f"    • deg({vertex}) = {self.degree(vertex)} (degree), neighbors: {neighbors}")
            print(f"    • w({vertex}) = {self.weighted_degree(vertex)} (weighted degree), edges: {weighted_neighbors}")
        print()
        
        print("Graph Theory Concepts:")
        print("  • Degree: deg(v) = number of edges incident to vertex v")
        print("  • Weighted degree: w(v) = sum of weights of edges incident to v")
        print("  • Total weight: w(G) = sum of all edge weights")
        print("  • In undirected graphs, each edge contributes to both vertices' degrees")
        
        # Handshaking lemma verification
        total_degree = sum(self.degree(v) for v in self._adjacency_list)
        print(f"\nHandshaking Lemma Verification:")
        print(f"  • Sum of all degrees: {total_degree}")
        print(f"  • 2 × |E| = {2 * info['edges']}")
        print(f"  • Handshaking lemma holds: {total_degree == 2 * info['edges']}")


def main():
    """Test the WeightedUndirectedGraph implementation."""
    print("=== Testing WeightedUndirectedGraph ===\n")
    
    # Test 1: Empty graph
    print("1. Creating empty weighted graph:")
    graph = WeightedUndirectedGraph()
    print(f"   {graph}")
    print(f"   Vertices: {graph.vertex_count()}, Edges: {graph.edge_count()}\n")
    
    # Test 2: Adding vertices and weighted edges
    print("2. Adding vertices and weighted edges:")
    graph.add_vertex('A')
    graph.add_vertex('B')
    graph.add_vertex('C')
    graph.add_edge('A', 'B', 5)
    graph.add_edge('B', 'C', 3)
    graph.add_edge('A', 'C', 8)
    print(f"   {graph}")
    print(f"   Total weight: {graph.total_weight()}\n")
    
    # Test 3: Creating weighted graph from data
    print("3. Creating weighted graph from data:")
    data = {
        'A': {'B': 5, 'C': 2},
        'B': {'A': 5, 'D': 1},
        'C': {'A': 2, 'D': 8, 'F': 10},
        'D': {'B': 1, 'C': 8, 'E': 6},
        'E': {'C': 4, 'D': 6},
        'F': {'C': 10},
        'G': {}  # Isolated vertex
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
        graph2.visualize("Weighted Network Graph", positions=house_positions)
    except ImportError:
        print("   Matplotlib or NetworkX not available for visualization")
    
    print("\n=== All weighted graph tests completed ===")
