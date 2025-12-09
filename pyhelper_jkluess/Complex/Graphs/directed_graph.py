from typing import List, Dict, Optional, Any, Set
try:
    from .graph import Graph
except ImportError:
    from graph import Graph

"""
Directed Graph Implementation

This module provides a directed graph that inherits from the base Graph class.
All common functionality is inherited; this class only provides convenience methods
and specific behavior for directed graphs.
"""


class DirectedGraph(Graph):
    """
    A directed graph implementation.
    Inherits all functionality from Graph with directed=True, weighted=False.
    """
    
    def __init__(self, data: Optional[Dict[Any, List[Any]]] = None):
        """
        Initialize a directed graph.
        
        Args:
            data: Optional dictionary where keys are vertices and values are lists of adjacent vertices
        """
        super().__init__(directed=True, weighted=False, data=data)
    
    # All core methods (add_vertex, remove_vertex, add_edge, remove_edge, has_vertex, 
    # has_edge, get_vertices, get_neighbors, get_predecessors, get_edges, vertex_count, 
    # edge_count) are inherited from Graph and work correctly for directed graphs.
    
    # Degree methods (degree, in_degree, out_degree) are inherited from Graph.
    
    # Path and reachability methods (find_path, is_reachable, is_simple_path, path_length) 
    # are inherited from Graph.
    
    # Cycle detection methods (has_cycle, find_cycles, is_acyclic) 
    # are inherited from Graph.
    
    # Connectivity methods (is_strongly_connected, get_strongly_connected_components) 
    # are inherited from Graph.
    
    # Adjacency matrix methods (get_adjacency_matrix) are inherited from Graph.
    
    # Visualization method (visualize) is inherited from Graph.
    
    def get_degree_sequence(self) -> Dict[str, List[int]]:
        """
        Get the degree sequences of the graph (in-degrees, out-degrees, and total degrees).
        
        Returns:
            Dict with 'in_degrees', 'out_degrees', and 'total_degrees' sorted in descending order
        """
        in_degrees = [self.in_degree(vertex) for vertex in self._adjacency_list]
        out_degrees = [self.out_degree(vertex) for vertex in self._adjacency_list]
        total_degrees = [self.degree(vertex) for vertex in self._adjacency_list]
        
        return {
            'in_degrees': sorted(in_degrees, reverse=True),
            'out_degrees': sorted(out_degrees, reverse=True),
            'total_degrees': sorted(total_degrees, reverse=True)
        }
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the directed graph structure.
        
        Returns:
            Dict containing various graph properties and statistics
        """
        if not self._adjacency_list:
            return {
                "vertices": 0,
                "edges": 0,
                "is_simple": True,
                "degree_sequences": {'in_degrees': [], 'out_degrees': [], 'total_degrees': []},
                "min_in_degree": 0,
                "max_in_degree": 0,
                "min_out_degree": 0,
                "max_out_degree": 0,
                "average_in_degree": 0.0,
                "average_out_degree": 0.0,
                "vertex_degrees": {}
            }
        
        vertex_degrees = {
            vertex: {
                'in_degree': self.in_degree(vertex),
                'out_degree': self.out_degree(vertex),
                'total_degree': self.degree(vertex)
            }
            for vertex in self._adjacency_list
        }
        
        in_degrees = [info['in_degree'] for info in vertex_degrees.values()]
        out_degrees = [info['out_degree'] for info in vertex_degrees.values()]
        
        return {
            "vertices": self.vertex_count(),
            "edges": self.edge_count(),
            "is_simple": self.is_simple_graph(),
            "degree_sequences": self.get_degree_sequence(),
            "min_in_degree": min(in_degrees) if in_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "min_out_degree": min(out_degrees) if out_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "average_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0.0,
            "average_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0.0,
            "vertex_degrees": vertex_degrees
        }
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: List[List[int]], vertices: Optional[List[Any]] = None) -> 'DirectedGraph':
        """
        Create a directed graph from an adjacency matrix.
        
        Args:
            matrix: n×n adjacency matrix where matrix[i][j] = 1 indicates a directed edge from i to j
            vertices: Optional list of vertex labels. If None, uses integers 0 to n-1
            
        Returns:
            New DirectedGraph instance
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
            for j in range(n):
                if matrix[i][j] == 1:
                    graph.add_edge(vertices[i], vertices[j])
        
        return graph
    
    def __str__(self) -> str:
        """String representation of the directed graph."""
        if not self._adjacency_list:
            return "Empty graph"
        
        result = "Directed Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = sorted(list(self._adjacency_list[vertex]))
            result += f"  {vertex} -> {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the directed graph."""
        return f"DirectedGraph(vertices={self.vertex_count()}, edges={self.edge_count()})"

    def print_graph_analysis(self):
        """
        Print a detailed analysis of the directed graph based on graph theory concepts.
        """
        info = self.get_graph_info()
        
        print("=== Directed Graph Theory Analysis ===")
        print(f"Directed Graph G = (V, E) with |V| = {info['vertices']} vertices and |E| = {info['edges']} directed edges")
        print()
        
        print("Basic Properties:")
        print(f"  • Simple graph (schlicht): {'Yes' if info['is_simple'] else 'No'}")
        print(f"  • Minimum in-degree: {info['min_in_degree']}")
        print(f"  • Maximum in-degree: {info['max_in_degree']}")
        print(f"  • Average in-degree: {info['average_in_degree']:.2f}")
        print(f"  • Minimum out-degree: {info['min_out_degree']}")
        print(f"  • Maximum out-degree: {info['max_out_degree']}")
        print(f"  • Average out-degree: {info['average_out_degree']:.2f}")
        print()
        
        print("Degree Sequences:")
        print(f"  • In-degree sequence: {info['degree_sequences']['in_degrees']}")
        print(f"  • Out-degree sequence: {info['degree_sequences']['out_degrees']}")
        print(f"  • Total degree sequence: {info['degree_sequences']['total_degrees']}")
        print()
        
        print("Individual Vertex Analysis:")
        for vertex in sorted(info['vertex_degrees'].keys()):
            degrees = info['vertex_degrees'][vertex]
            successors = sorted(self.get_neighbors(vertex))
            predecessors = sorted(self.get_predecessors(vertex))
            print(f"  Vertex {vertex}:")
            print(f"    • deg⁺({vertex}) = {degrees['out_degree']} (out-degree), successors: {successors}")
            print(f"    • deg⁻({vertex}) = {degrees['in_degree']} (in-degree), predecessors: {predecessors}")
            print(f"    • deg({vertex}) = {degrees['total_degree']} (total degree)")
        print()
        
        print("Graph Theory Concepts:")
        print("  • Out-degree: deg⁺(v) = number of edges leaving vertex v")
        print("  • In-degree: deg⁻(v) = number of edges entering vertex v")
        print("  • Total degree: deg(v) = deg⁺(v) + deg⁻(v)")
        print("  • Predecessor: Vertex u is predecessor of v if edge (u,v) exists")
        print("  • Successor: Vertex w is successor of v if edge (v,w) exists")
        print("  • Simple: No self-loops (edges from vertex to itself)")


def main():
    """Test the DirectedGraph implementation."""
    print("=== Testing DirectedGraph ===\n")
    
    # Create directed graph
    print("Creating directed graph:")
    data = {
        'X': ['Y', 'Z'],
        'Y': ['Z'],
        'Z': ['X']
    }
    graph = DirectedGraph(data)
    print(f"{graph}\n")
    
    # Test graph analysis
    print("Graph Analysis:")
    graph.print_graph_analysis()
    
    print("\n=== Tests completed ===")
