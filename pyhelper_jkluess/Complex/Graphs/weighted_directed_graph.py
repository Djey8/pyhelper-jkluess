from typing import List, Dict, Optional, Any, Union
try:
    from .graph import Graph
except ImportError:
    from graph import Graph

"""
Weighted Directed Graph Implementation

This module provides a weighted directed graph that inherits from the base Graph class.
All common functionality is inherited; this class only provides convenience methods
and specific behavior for weighted directed graphs.
"""


class WeightedDirectedGraph(Graph):
    """
    A weighted directed graph implementation.
    Inherits all functionality from Graph with directed=True, weighted=True.
    """
    
    def __init__(self, data: Optional[Dict[Any, List[tuple]]] = None):
        """
        Initialize a weighted directed graph.
        
        Args:
            data: Optional dictionary where keys are vertices and values are lists of tuples (neighbor, weight)
        """
        super().__init__(directed=True, weighted=True, data=data)
    
    # All core methods (add_vertex, remove_vertex, add_edge, remove_edge, has_vertex, 
    # has_edge, get_vertices, get_neighbors, get_predecessors, get_edges, vertex_count, 
    # edge_count, update_edge_weight, get_edge_weight, get_weighted_neighbors, 
    # get_weighted_predecessors) are inherited from Graph and work correctly for 
    # weighted directed graphs.
    
    # Degree methods (degree, in_degree, out_degree, weighted_degree, weighted_in_degree, 
    # weighted_out_degree) are inherited from Graph.
    
    # Path and reachability methods (find_path, is_reachable, is_simple_path, path_length,
    # path_weight) are inherited from Graph.
    
    # Cycle detection methods (has_cycle, find_cycles, is_acyclic) 
    # are inherited from Graph.
    
    # Connectivity methods (is_strongly_connected, get_strongly_connected_components) 
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
    
    def get_weighted_degree_sequence(self) -> Dict[str, List[Union[int, float]]]:
        """
        Get the weighted degree sequences (in-degrees, out-degrees, and total degrees).
        
        Returns:
            Dict with 'weighted_in_degrees', 'weighted_out_degrees', and 'weighted_total_degrees'
        """
        weighted_in = [self.weighted_in_degree(vertex) for vertex in self._adjacency_list]
        weighted_out = [self.weighted_out_degree(vertex) for vertex in self._adjacency_list]
        weighted_total = [self.weighted_degree(vertex) for vertex in self._adjacency_list]
        
        return {
            'weighted_in_degrees': sorted(weighted_in, reverse=True),
            'weighted_out_degrees': sorted(weighted_out, reverse=True),
            'weighted_total_degrees': sorted(weighted_total, reverse=True)
        }
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the weighted directed graph structure.
        
        Returns:
            Dict containing various graph properties and statistics
        """
        if not self._adjacency_list:
            return {
                "vertices": 0,
                "edges": 0,
                "total_weight": 0,
                "is_simple": True,
                "degree_sequences": {'in_degrees': [], 'out_degrees': [], 'total_degrees': []},
                "weighted_degree_sequences": {
                    'weighted_in_degrees': [],
                    'weighted_out_degrees': [],
                    'weighted_total_degrees': []
                },
                "min_in_degree": 0,
                "max_in_degree": 0,
                "min_out_degree": 0,
                "max_out_degree": 0,
                "average_in_degree": 0.0,
                "average_out_degree": 0.0,
                "min_weighted_in_degree": 0,
                "max_weighted_in_degree": 0,
                "min_weighted_out_degree": 0,
                "max_weighted_out_degree": 0,
                "average_weighted_in_degree": 0.0,
                "average_weighted_out_degree": 0.0,
                "vertex_degrees": {},
                "weight_statistics": {}
            }
        
        vertex_degrees = {
            vertex: {
                'in_degree': self.in_degree(vertex),
                'out_degree': self.out_degree(vertex),
                'total_degree': self.degree(vertex),
                'weighted_in_degree': self.weighted_in_degree(vertex),
                'weighted_out_degree': self.weighted_out_degree(vertex),
                'weighted_total_degree': self.weighted_degree(vertex)
            }
            for vertex in self._adjacency_list
        }
        
        in_degrees = [info['in_degree'] for info in vertex_degrees.values()]
        out_degrees = [info['out_degree'] for info in vertex_degrees.values()]
        weighted_in = [info['weighted_in_degree'] for info in vertex_degrees.values()]
        weighted_out = [info['weighted_out_degree'] for info in vertex_degrees.values()]
        
        return {
            "vertices": self.vertex_count(),
            "edges": self.edge_count(),
            "total_weight": self.total_weight(),
            "is_simple": self.is_simple_graph(),
            "degree_sequences": self.get_degree_sequence(),
            "weighted_degree_sequences": self.get_weighted_degree_sequence(),
            "min_in_degree": min(in_degrees) if in_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "min_out_degree": min(out_degrees) if out_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "average_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0.0,
            "average_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0.0,
            "min_weighted_in_degree": min(weighted_in) if weighted_in else 0,
            "max_weighted_in_degree": max(weighted_in) if weighted_in else 0,
            "min_weighted_out_degree": min(weighted_out) if weighted_out else 0,
            "max_weighted_out_degree": max(weighted_out) if weighted_out else 0,
            "average_weighted_in_degree": sum(weighted_in) / len(weighted_in) if weighted_in else 0.0,
            "average_weighted_out_degree": sum(weighted_out) / len(weighted_out) if weighted_out else 0.0,
            "vertex_degrees": vertex_degrees,
            "weight_statistics": self.get_weight_statistics()
        }
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: List[List[Union[int, float]]], 
                            vertices: Optional[List[Any]] = None) -> 'WeightedDirectedGraph':
        """
        Create a weighted directed graph from an adjacency matrix.
        
        Args:
            matrix: n×n adjacency matrix where matrix[i][j] is the weight from i to j (0 means no edge)
            vertices: Optional list of vertex labels. If None, uses integers 0 to n-1
            
        Returns:
            New WeightedDirectedGraph instance
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
        
        # Add edges with weights from matrix
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    graph.add_edge(vertices[i], vertices[j], matrix[i][j])
        
        return graph
    
    def __str__(self) -> str:
        """String representation of the weighted directed graph."""
        if not self._adjacency_list:
            return "Empty graph"
        
        result = "Weighted Directed Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = sorted([(neighbor, weight) for neighbor, weight in self._adjacency_list[vertex].items()])
            result += f"  {vertex} -> {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the weighted directed graph."""
        return f"WeightedDirectedGraph(vertices={self.vertex_count()}, edges={self.edge_count()}, total_weight={self.total_weight()})"

    def print_graph_analysis(self):
        """
        Print a detailed analysis of the weighted directed graph based on graph theory concepts.
        """
        info = self.get_graph_info()
        
        print("=== Weighted Directed Graph Theory Analysis ===")
        print(f"Weighted Directed Graph G = (V, E, w) with |V| = {info['vertices']} vertices and |E| = {info['edges']} weighted directed edges")
        print(f"Total weight w(G) = {info['total_weight']}")
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
        
        print("Weighted Degree Properties:")
        print(f"  • Minimum weighted in-degree: {info['min_weighted_in_degree']}")
        print(f"  • Maximum weighted in-degree: {info['max_weighted_in_degree']}")
        print(f"  • Average weighted in-degree: {info['average_weighted_in_degree']:.2f}")
        print(f"  • Minimum weighted out-degree: {info['min_weighted_out_degree']}")
        print(f"  • Maximum weighted out-degree: {info['max_weighted_out_degree']}")
        print(f"  • Average weighted out-degree: {info['average_weighted_out_degree']:.2f}")
        print()
        
        print("Degree Sequences:")
        print(f"  • In-degree sequence: {info['degree_sequences']['in_degrees']}")
        print(f"  • Out-degree sequence: {info['degree_sequences']['out_degrees']}")
        print(f"  • Total degree sequence: {info['degree_sequences']['total_degrees']}")
        print()
        
        print("Weighted Degree Sequences:")
        print(f"  • Weighted in-degree sequence: {info['weighted_degree_sequences']['weighted_in_degrees']}")
        print(f"  • Weighted out-degree sequence: {info['weighted_degree_sequences']['weighted_out_degrees']}")
        print(f"  • Weighted total degree sequence: {info['weighted_degree_sequences']['weighted_total_degrees']}")
        print()
        
        print("Weight Statistics:")
        stats = info['weight_statistics']
        if stats:
            print(f"  • Minimum edge weight: {stats.get('min_weight', 'N/A')}")
            print(f"  • Maximum edge weight: {stats.get('max_weight', 'N/A')}")
            print(f"  • Average edge weight: {stats.get('average_weight', 0):.2f}")
        print()
        
        print("Individual Vertex Analysis:")
        for vertex in sorted(info['vertex_degrees'].keys()):
            degrees = info['vertex_degrees'][vertex]
            successors = sorted(self.get_neighbors(vertex))
            predecessors = sorted(self.get_predecessors(vertex))
            weighted_successors = sorted(self.get_weighted_neighbors(vertex))
            weighted_predecessors = sorted(self.get_weighted_predecessors(vertex))
            print(f"  Vertex {vertex}:")
            print(f"    • deg⁺({vertex}) = {degrees['out_degree']} (out-degree), successors: {successors}")
            print(f"    • deg⁻({vertex}) = {degrees['in_degree']} (in-degree), predecessors: {predecessors}")
            print(f"    • deg({vertex}) = {degrees['total_degree']} (total degree)")
            print(f"    • w⁺({vertex}) = {degrees['weighted_out_degree']} (weighted out-degree), edges: {weighted_successors}")
            print(f"    • w⁻({vertex}) = {degrees['weighted_in_degree']} (weighted in-degree), edges: {weighted_predecessors}")
            print(f"    • w({vertex}) = {degrees['weighted_total_degree']} (weighted total degree)")
        print()
        
        print("Graph Theory Concepts:")
        print("  • Out-degree: deg⁺(v) = number of edges leaving vertex v")
        print("  • In-degree: deg⁻(v) = number of edges entering vertex v")
        print("  • Total degree: deg(v) = deg⁺(v) + deg⁻(v)")
        print("  • Weighted out-degree: w⁺(v) = sum of weights of edges leaving v")
        print("  • Weighted in-degree: w⁻(v) = sum of weights of edges entering v")
        print("  • Weighted total degree: w(v) = w⁺(v) + w⁻(v)")
        print("  • Total weight: w(G) = sum of all edge weights")


def main():
    """Test the WeightedDirectedGraph implementation with various graph structures."""
    print("=== Testing WeightedDirectedGraph Implementation ===\n")
    
    # Test 1: Empty graph
    print("1. Creating empty weighted graph:")
    graph = WeightedDirectedGraph()
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
    print(f"   Weighted edges: {graph.get_edges()}\n")
    
    # Test 3: Creating a simple weighted directed graph
    print("3. Creating a simple weighted directed graph:")
    simple_data = {
        'X': [('Y', 2), ('Z', 4)],
        'Y': [('Z', 1)],
        'Z': [('X', 3)]
    }
    simple_graph = WeightedDirectedGraph(simple_data)
    print(f"   {simple_graph}")
    print(f"   Vertices: {simple_graph.get_vertices()}")
    print(f"   Total edges: {simple_graph.edge_count()}")
    print(f"   Total weight: {simple_graph.total_weight()}\n")
    
    # Test 4: Testing weighted degree and graph theory features
    print("4. Testing weighted degree and graph theory features:")
    print(f"   Out-degree of X: {simple_graph.out_degree('X')}")
    print(f"   Weighted out-degree of X: {simple_graph.weighted_out_degree('X')}")
    print(f"   In-degree of Z: {simple_graph.in_degree('Z')}")
    print(f"   Weighted in-degree of Z: {simple_graph.weighted_in_degree('Z')}")
    print(f"   Total degree of Y: {simple_graph.degree('Y')}")
    print(f"   Weighted total degree of Y: {simple_graph.weighted_degree('Y')}")
    print(f"   Degree sequences: {simple_graph.get_degree_sequence()}")
    print(f"   Weighted degree sequences: {simple_graph.get_weighted_degree_sequence()}")
    print(f"   Is simple graph: {simple_graph.is_simple_graph()}\n")
    
    # Test 5: Comprehensive weighted graph analysis
    print("5. Comprehensive weighted directed graph analysis:")
    simple_graph.print_graph_analysis()
    print()
    
    # Test 6: Testing with self-loop (non-simple graph)
    print("6. Testing with self-loop:")
    loop_graph = WeightedDirectedGraph({'A': [('B', 2), ('A', 1)], 'B': [('A', 3)]})  # A has weighted self-loop
    print(f"   Graph with weighted self-loop: {loop_graph}")
    print(f"   Is simple: {loop_graph.is_simple_graph()}")
    print(f"   Out-degree of A (with self-loop): {loop_graph.out_degree('A')}")
    print(f"   Weighted out-degree of A (with self-loop): {loop_graph.weighted_out_degree('A')}")
    print(f"   In-degree of A (with self-loop): {loop_graph.in_degree('A')}")
    print(f"   Weighted in-degree of A (with self-loop): {loop_graph.weighted_in_degree('A')}\n")
    
    # Test 7: Testing removal and weight update operations
    print("7. Testing removal and weight update operations:")
    print(f"   Before removal: {simple_graph}")
    print(f"   Weight of Y->Z before: {simple_graph.get_edge_weight('Y', 'Z')}")
    simple_graph.update_edge_weight('Y', 'Z', 5)
    print(f"   After updating Y->Z weight to 5: {simple_graph.get_edge_weight('Y', 'Z')}")
    simple_graph.remove_edge('Y', 'Z')
    print(f"   After removing Y->Z: {simple_graph}")
    simple_graph.add_edge('Y', 'Z', 1)
    print(f"   After adding back Y->Z with weight 1: {simple_graph}\n")
    
    # Test 8: Testing weighted predecessors and successors
    print("8. Testing weighted predecessors and successors:")
    print(f"   Weighted successors of X (outgoing): {simple_graph.get_weighted_neighbors('X')}")
    print(f"   Weighted predecessors of X (incoming): {simple_graph.get_weighted_predecessors('X')}")
    print(f"   Weighted successors of Z: {simple_graph.get_weighted_neighbors('Z')}")
    print(f"   Weighted predecessors of Z: {simple_graph.get_weighted_predecessors('Z')}\n")
    
    # Test 9: Weighted graph information summary
    print("9. Weighted directed graph information summary:")
    info = simple_graph.get_graph_info()
    for key, value in info.items():
        if key != 'vertex_degrees':  # Skip detailed vertex info for summary
            print(f"   {key}: {value}")
    print()
    
    # Test 10: Weighted House of Nikolaus visualization
    print("10. Creating and visualizing the Weighted House of Nikolaus:")
    house_data = {
        'A': [('B', 1), ('D', 7)],    # Bottom left -> bottom right (weight 2), top left (weight 3)
        'B': [('C', 2)],              # Bottom right -> top right (weight 2)
        'C': [('E', 3), ('A', 6)],    # Top left -> roof peak (weight 1), bottom left (weight 3)
        'D': [('C', 5), ('B', 8)],    # Top right -> top left (weight 2), bottom right (weight 3)
        'E': [('D', 4)]               # Roof peak -> top right (weight 1)
    }
    house_graph = WeightedDirectedGraph(house_data)
    print(f"   {house_graph}")
    print(f"   Total weight of all paths: {house_graph.total_weight()}")
    print(f"   Weight statistics: {house_graph.get_weight_statistics()}")
    
    try:
        house_positions = {
            'A': (0, 0),    # Bottom left corner
            'B': (2, 0),    # Bottom right corner  
            'C': (0, 2),    # Top left corner
            'D': (2, 2),    # Top right corner
            'E': (1, 3)     # Roof peak
        }
        house_graph.visualize("Weighted House of Nikolaus (Directed)", 
                            figsize=(12, 10), 
                            positions=house_positions)
    except ImportError:
        print("   Matplotlib or NetworkX not available for visualization")
    
    print(f"\n   Weighted House of Nikolaus has {house_graph.vertex_count()} vertices and {house_graph.edge_count()} weighted directed edges")
    
    print("\n=== WeightedDirectedGraph tests completed ===")
