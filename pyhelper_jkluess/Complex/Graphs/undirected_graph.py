from typing import List, Dict, Optional, Any
try:
    from .graph import Graph
except ImportError:
    from graph import Graph

"""
Undirected Graph Implementation

This module provides an undirected graph that inherits from the base Graph class.
All common functionality is inherited; this class only provides convenience methods
and specific behavior for undirected graphs.
"""


class UndirectedGraph(Graph):
    """
    An undirected graph implementation.
    Inherits all functionality from Graph with directed=False, weighted=False.
    """
    
    def __init__(self, data: Optional[Dict[Any, List[Any]]] = None):
        """
        Initialize an undirected graph.
        
        Args:
            data: Optional dictionary where keys are vertices and values are lists of adjacent vertices
        """
        super().__init__(directed=False, weighted=False, data=data)
    
    # All core methods (add_vertex, remove_vertex, add_edge, remove_edge, has_vertex, 
    # has_edge, get_vertices, get_neighbors, get_edges, vertex_count, edge_count)
    # are inherited from Graph and work correctly for undirected graphs.
    
    # Path and reachability methods (find_path, is_reachable, is_simple_path, path_length) 
    # are inherited from Graph.
    
    # Cycle detection methods (has_cycle, find_cycles, is_acyclic) 
    # are inherited from Graph.
    
    # Connectivity methods (is_connected, get_connected_components) 
    # are inherited from Graph.
    
    # Adjacency matrix methods (get_adjacency_matrix) are inherited from Graph.
    
    # Visualization method (visualize) is inherited from Graph.
    
    # Degree method (degree) is inherited from Graph.
    
    def get_degree_sequence(self) -> List[int]:
        """
        Get the degree sequence of the graph (degrees of all vertices sorted in descending order).
        
        Returns:
            List[int]: Sorted list of vertex degrees in descending order
        """
        degrees = [self.degree(vertex) for vertex in self._adjacency_list]
        return sorted(degrees, reverse=True)
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the graph structure.
        
        Returns:
            Dict containing various graph properties and statistics
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
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: List[List[int]], vertices: Optional[List[Any]] = None) -> 'UndirectedGraph':
        """
        Create an undirected graph from an adjacency matrix.
        
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
        
        # Add edges from matrix (only check upper triangle for undirected graph)
        for i in range(n):
            for j in range(i, n):
                if matrix[i][j] == 1:
                    graph.add_edge(vertices[i], vertices[j])
        
        return graph
    
    def __str__(self) -> str:
        """String representation of the undirected graph."""
        if not self._adjacency_list:
            return "Empty graph"
        
        result = "Undirected Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = sorted(list(self._adjacency_list[vertex]))
            result += f"  {vertex}: {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the undirected graph."""
        return f"UndirectedGraph(vertices={self.vertex_count()}, edges={self.edge_count()})"
    
    def print_graph_analysis(self):
        """
        Print a detailed analysis of the undirected graph based on graph theory concepts.
        
        Provides undirected-specific formatting while using base Graph functionality.
        """
        info = self.get_graph_info()
        
        print("=== Undirected Graph Theory Analysis ===")
        print(f"Undirected Graph G = (V, E) with |V| = {info['vertices']} vertices and |E| = {info['edges']} edges")
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
        
        print("Graph Theory Properties:")
        print(f"  • Connected: {self.is_connected()}")
        if not self.is_connected():
            components = self.get_connected_components()
            print(f"  • Number of connected components: {len(components)}")
        print(f"  • Has cycles: {self.has_cycle()}")
        print(f"  • Is acyclic (tree/forest): {self.is_acyclic()}")
        print()
        
        print("Handshaking Lemma Verification:")
        sum_degrees = sum(info['degree_sequence'])
        print(f"  • Sum of all degrees: {sum_degrees}")
        print(f"  • 2 × |E| = {2 * info['edges']}")
        print(f"  • Handshaking lemma holds: {sum_degrees == 2 * info['edges']}")



def main():
    """Test the UndirectedGraph implementation."""
    print("=== Testing UndirectedGraph ===\n")
    
    # Create graph with initial data
    print("Creating graph with initial data:")
    data = {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'D'],
        'C': ['A', 'D', 'E'],
        'D': ['B', 'C', 'E'],
        'E': ['C', 'D'],
        'L': []
    }
    graph = UndirectedGraph(data)
    print(f"{graph}\n")
    
    # Test graph analysis
    print("Graph Analysis:")
    graph.print_graph_analysis()
    
    print("\n=== Tests completed ===")