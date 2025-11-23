import networkx as nx
from typing import List, Set, Dict, Optional, Any

"""
Directed Graph Implementation

This module provides a basic directed graph data structure with common operations
like adding/removing vertices and edges, and visualization capabilities.
"""

import matplotlib.pyplot as plt


class DirectedGraph:
    """
    A basic directed graph implementation using adjacency list representation.
    """
    
    def __init__(self, data: Optional[Dict[Any, List[Any]]] = None):
        """
        Initialize a directed graph.
        
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
        
        # Remove all edges pointing to this vertex
        for v in self._adjacency_list:
            self._adjacency_list[v].discard(vertex)
        
        # Remove the vertex itself
        del self._adjacency_list[vertex]
        return True
    
    def add_edge(self, from_vertex: Any, to_vertex: Any) -> bool:
        """
        Add a directed edge from one vertex to another.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            
        Returns:
            bool: True if edge was added, False if edge already exists
        """
        if from_vertex not in self._adjacency_list:
            self.add_vertex(from_vertex)
        if to_vertex not in self._adjacency_list:
            self.add_vertex(to_vertex)
        
        if to_vertex not in self._adjacency_list[from_vertex]:
            self._adjacency_list[from_vertex].add(to_vertex)
            return True
        return False
    
    def remove_edge(self, from_vertex: Any, to_vertex: Any) -> bool:
        """
        Remove a directed edge between two vertices.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            
        Returns:
            bool: True if edge was removed, False if edge doesn't exist
        """
        if (from_vertex in self._adjacency_list and
            to_vertex in self._adjacency_list[from_vertex]):
            
            self._adjacency_list[from_vertex].discard(to_vertex)
            return True
        return False
    
    def has_vertex(self, vertex: Any) -> bool:
        """Check if a vertex exists in the graph."""
        return vertex in self._adjacency_list
    
    def has_edge(self, from_vertex: Any, to_vertex: Any) -> bool:
        """Check if a directed edge exists from one vertex to another."""
        return (from_vertex in self._adjacency_list and 
                to_vertex in self._adjacency_list[from_vertex])
    
    def get_vertices(self) -> List[Any]:
        """Get all vertices in the graph."""
        return list(self._adjacency_list.keys())
    
    def get_neighbors(self, vertex: Any) -> List[Any]:
        """Get all outgoing neighbors of a vertex."""
        if vertex in self._adjacency_list:
            return list(self._adjacency_list[vertex])
        return []
    
    def get_predecessors(self, vertex: Any) -> List[Any]:
        """
        Get all incoming neighbors (predecessors) of a vertex.
        
        Args:
            vertex: The vertex to get predecessors for
            
        Returns:
            List of vertices that have edges pointing to the given vertex
        """
        if vertex not in self._adjacency_list:
            return []
        
        predecessors = []
        for v in self._adjacency_list:
            if vertex in self._adjacency_list[v]:
                predecessors.append(v)
        return predecessors
    
    def in_degree(self, vertex: Any) -> int:
        """
        Get the in-degree of a vertex (number of incoming edges).
        
        Args:
            vertex: The vertex to get in-degree for
            
        Returns:
            Number of edges pointing to the vertex
        """
        return len(self.get_predecessors(vertex))
    
    def out_degree(self, vertex: Any) -> int:
        """
        Get the out-degree of a vertex (number of outgoing edges).
        
        Args:
            vertex: The vertex to get out-degree for
            
        Returns:
            Number of edges pointing from the vertex
        """
        if vertex not in self._adjacency_list:
            return 0
        return len(self._adjacency_list[vertex])
    
    def get_edges(self) -> List[tuple]:
        """Get all directed edges in the graph."""
        edges = []
        for vertex in self._adjacency_list:
            for neighbor in self._adjacency_list[vertex]:
                edges.append((vertex, neighbor))
        return edges
    
    def vertex_count(self) -> int:
        """Get the number of vertices in the graph."""
        return len(self._adjacency_list)
    
    def edge_count(self) -> int:
        """Get the number of directed edges in the graph."""
        return len(self.get_edges())
    
    def visualize(self, title: str = "Directed Graph", figsize: tuple = (10, 8), positions: Optional[Dict[Any, tuple]] = None):
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
        
        # Create NetworkX directed graph
        G = nx.DiGraph()
        
        # Add vertices
        for vertex in self._adjacency_list:
            G.add_node(vertex)
        
        # Add directed edges
        for from_vertex, to_vertex in self.get_edges():
            G.add_edge(from_vertex, to_vertex)
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Use custom positions if provided, otherwise use spring layout
        if positions:
            pos = positions
        else:
            pos = nx.spring_layout(G, seed=42)
        
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='lightcoral',
                node_size=500,
                font_size=12,
                font_weight='bold',
                edge_color='gray',
                width=2,
                arrows=True,
                arrowsize=20,
                arrowstyle='->')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def degree(self, vertex: Any) -> int:
        """
        Get the total degree of a vertex (in-degree + out-degree).
        
        For directed graphs, the total degree is the sum of incoming and outgoing edges.
        
        Args:
            vertex: The vertex to get the degree for
            
        Returns:
            int: The total degree of the vertex, or 0 if vertex doesn't exist
            
        Theory:
            In a directed graph G = (V, E), the degree of a vertex v is:
            deg(v) = deg⁺(v) + deg⁻(v)
            where deg⁺(v) is out-degree and deg⁻(v) is in-degree
        """
        if vertex not in self._adjacency_list:
            return 0
        return self.in_degree(vertex) + self.out_degree(vertex)

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

    def is_simple_graph(self) -> bool:
        """
        Check if the graph is simple (schlicht).
        
        A simple directed graph has no self-loops (edges from a vertex to itself).
        
        Returns:
            bool: True if the graph is simple, False otherwise
            
        Theory:
            A directed graph is called simple (schlicht) if it contains no edge whose
            start and end nodes are identical (no self-loops).
        """
        for vertex in self._adjacency_list:
            if vertex in self._adjacency_list[vertex]:
                return False
        return True

    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the directed graph structure.
        
        Returns:
            Dict containing various graph properties and statistics
            
        Theory:
            For a directed graph G = (V, E):
            - Vertices (V): Set of nodes in the graph
            - Edges (E): Set of directed connections between vertices
            - Out-degree: deg⁺(v) = number of edges leaving vertex v
            - In-degree: deg⁻(v) = number of edges entering vertex v
            - Total degree: deg(v) = deg⁺(v) + deg⁻(v)
            - Simple graph: Contains no self-loops
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

    def print_graph_analysis(self):
        """
        Print a detailed analysis of the directed graph based on graph theory concepts.
        
        Theory explanation:
        - Directed Graph G = (V, E) where V is vertices and E is directed edges
        - Out-degree: deg⁺(v) = number of edges leaving vertex v
        - In-degree: deg⁻(v) = number of edges entering vertex v
        - Total degree: deg(v) = deg⁺(v) + deg⁻(v)
        - Predecessor: A vertex u is a predecessor of v if edge (u,v) exists
        - Successor: A vertex w is a successor of v if edge (v,w) exists
        - Simple graph: No self-loops (edges from vertex to itself)
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

    # Path and reachability methods
    def find_path(self, start: Any, end: Any) -> Optional[List[Any]]:
        """
        Find a path from start vertex to end vertex using BFS.
        
        Theory: A path (Weg/Pfad) from v to v' is a sequence of vertices v₀, v₁, ..., vₖ where:
        - v₀ = v (start vertex)
        - vₖ = v' (end vertex)
        - (vᵢ, vᵢ₊₁) ∈ E for i = 0, ..., k-1 (directed edges)
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
        Check if end vertex is reachable from start vertex following directed edges.
        
        Theory: Vertex v is reachable from vertex u if there exists a directed path from u to v.
        
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
        Check if the graph contains any cycle (using DFS).
        
        Theory: A cycle (Zyklus) is a path v₀, v₁, ..., vₖ where v₀ = vₖ.
        A circle (Kreis) is a cycle where v₀, v₁, ..., vₖ₋₁ are pairwise distinct.
        Trivial cycles (length 1 or 2) are often not considered.
        A graph without cycles is called acyclic (azyklisch).
        For directed graphs, this is called a DAG (Directed Acyclic Graph).
        
        Returns:
            True if the graph contains a cycle, False otherwise
        """
        if not self._adjacency_list:
            return False
        
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in self._adjacency_list}
        
        def dfs_has_cycle(vertex: Any) -> bool:
            color[vertex] = GRAY
            
            for neighbor in self._adjacency_list[vertex]:
                if color[neighbor] == GRAY:  # Back edge found
                    return True
                if color[neighbor] == WHITE and dfs_has_cycle(neighbor):
                    return True
            
            color[vertex] = BLACK
            return False
        
        for vertex in self._adjacency_list:
            if color[vertex] == WHITE:
                if dfs_has_cycle(vertex):
                    return True
        
        return False
    
    def find_cycles(self) -> List[List[Any]]:
        """
        Find all simple cycles in the directed graph using DFS.
        
        Theory: Returns all simple cycles (Kreise) where vertices are pairwise distinct.
        
        Returns:
            List of cycles, where each cycle is a list of vertices
        """
        cycles = []
        visited = set()
        path_stack = []
        path_set = set()
        
        def dfs_find_cycles(vertex: Any):
            visited.add(vertex)
            path_stack.append(vertex)
            path_set.add(vertex)
            
            for neighbor in self._adjacency_list[vertex]:
                if neighbor in path_set:
                    # Found a cycle
                    cycle_start = path_stack.index(neighbor)
                    cycle = path_stack[cycle_start:]
                    if cycle not in cycles:
                        cycles.append(cycle[:])
                elif neighbor not in visited:
                    dfs_find_cycles(neighbor)
            
            path_stack.pop()
            path_set.remove(vertex)
        
        for vertex in self._adjacency_list:
            if vertex not in visited:
                dfs_find_cycles(vertex)
        
        return cycles
    
    def is_acyclic(self) -> bool:
        """
        Check if the graph is acyclic (DAG - Directed Acyclic Graph).
        
        Theory: An acyclic directed graph contains no cycles. Such graphs are called DAGs.
        
        Returns:
            True if the graph is acyclic, False otherwise
        """
        return not self.has_cycle()
    
    # Connectivity methods
    def is_strongly_connected(self) -> bool:
        """
        Check if the graph is strongly connected.
        
        Theory: A directed graph G is strongly connected (stark zusammenhängend) if every
        vertex is reachable from every other vertex following directed edges.
        
        Returns:
            True if the graph is strongly connected, False otherwise
        """
        if not self._adjacency_list:
            return True
        
        # Check if all vertices are reachable from first vertex
        first_vertex = next(iter(self._adjacency_list))
        
        # Forward DFS from first vertex
        visited = set()
        stack = [first_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            stack.extend(self._adjacency_list[vertex] - visited)
        
        if len(visited) != len(self._adjacency_list):
            return False
        
        # Reverse DFS: check if all vertices can reach first vertex
        # Create reverse graph
        reverse_adj = {v: set() for v in self._adjacency_list}
        for vertex in self._adjacency_list:
            for neighbor in self._adjacency_list[vertex]:
                reverse_adj[neighbor].add(vertex)
        
        visited = set()
        stack = [first_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            stack.extend(reverse_adj[vertex] - visited)
        
        return len(visited) == len(self._adjacency_list)
    
    def get_strongly_connected_components(self) -> List[Set[Any]]:
        """
        Get all strongly connected components using Kosaraju's algorithm.
        
        Theory: A strongly connected component is a maximal subgraph where every vertex
        is reachable from every other vertex. This represents equivalence classes of
        vertices with respect to the "mutually reachable" relation.
        
        Returns:
            List of sets, where each set contains vertices in a strongly connected component
        """
        if not self._adjacency_list:
            return []
        
        # Step 1: Fill order using DFS
        visited = set()
        finish_order = []
        
        def dfs_fill_order(vertex: Any):
            visited.add(vertex)
            for neighbor in self._adjacency_list[vertex]:
                if neighbor not in visited:
                    dfs_fill_order(neighbor)
            finish_order.append(vertex)
        
        for vertex in self._adjacency_list:
            if vertex not in visited:
                dfs_fill_order(vertex)
        
        # Step 2: Create reverse graph
        reverse_adj = {v: set() for v in self._adjacency_list}
        for vertex in self._adjacency_list:
            for neighbor in self._adjacency_list[vertex]:
                reverse_adj[neighbor].add(vertex)
        
        # Step 3: DFS on reverse graph in reverse finish order
        visited = set()
        components = []
        
        def dfs_component(vertex: Any, component: Set[Any]):
            visited.add(vertex)
            component.add(vertex)
            for neighbor in reverse_adj[vertex]:
                if neighbor not in visited:
                    dfs_component(neighbor, component)
        
        for vertex in reversed(finish_order):
            if vertex not in visited:
                component = set()
                dfs_component(vertex, component)
                components.append(component)
        
        return components
    
    # Adjacency matrix methods
    def get_adjacency_matrix(self) -> List[List[int]]:
        """
        Get the adjacency matrix representation of the directed graph.
        
        Theory: For directed graph G = (V, E) with V = {v₁, ..., vₙ}, the adjacency
        matrix A ∈ ℝⁿˣⁿ is defined as:
        - aᵢⱼ = 1 if directed edge from vertex vᵢ to vⱼ exists
        - aᵢⱼ = 0 if no directed edge from vertex vᵢ to vⱼ exists
        
        For directed graphs, the matrix is generally not symmetric.
        
        Returns:
            n×n matrix (list of lists) where matrix[i][j] = 1 if edge (i,j) exists, 0 otherwise
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
    def from_adjacency_matrix(cls, matrix: List[List[int]], vertices: Optional[List[Any]] = None) -> 'DirectedGraph':
        """
        Create a directed graph from an adjacency matrix.
        
        Theory: Converts an adjacency matrix representation back to a directed graph structure.
        
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
        """String representation of the graph."""
        if not self._adjacency_list:
            return "Empty graph"
        
        result = "Directed Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = sorted(list(self._adjacency_list[vertex]))
            result += f"  {vertex} -> {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the graph."""
        return f"DirectedGraph(vertices={self.vertex_count()}, edges={self.edge_count()})"


def main():
    """Test the DirectedGraph implementation with various graph structures."""
    print("=== Testing DirectedGraph Implementation ===\n")
    
    # Test 1: Empty graph
    print("1. Creating empty graph:")
    graph = DirectedGraph()
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
    
    # Test 3: Creating a simple directed graph
    print("3. Creating a simple directed graph:")
    simple_data = {
        'X': ['Y', 'Z'],
        'Y': ['Z'],
        'Z': ['X']
    }
    simple_graph = DirectedGraph(simple_data)
    print(f"   {simple_graph}")
    print(f"   Vertices: {simple_graph.get_vertices()}")
    print(f"   Total edges: {simple_graph.edge_count()}\n")
    
    # Test 4: Testing degree and graph theory features
    print("4. Testing degree and graph theory features:")
    print(f"   Out-degree of X: {simple_graph.out_degree('X')}")
    print(f"   In-degree of Z: {simple_graph.in_degree('Z')}")
    print(f"   Total degree of Y: {simple_graph.degree('Y')}")
    print(f"   Degree sequences: {simple_graph.get_degree_sequence()}")
    print(f"   Is simple graph: {simple_graph.is_simple_graph()}\n")
    
    # Test 5: Comprehensive graph analysis
    print("5. Comprehensive directed graph analysis:")
    simple_graph.print_graph_analysis()
    print()
    
    # Test 6: Testing with self-loop (non-simple graph)
    print("6. Testing with self-loop:")
    loop_graph = DirectedGraph({'A': ['B', 'A'], 'B': ['A']})  # A has self-loop
    print(f"   Graph with self-loop: {loop_graph}")
    print(f"   Is simple: {loop_graph.is_simple_graph()}")
    print(f"   Out-degree of A (with self-loop): {loop_graph.out_degree('A')}")
    print(f"   In-degree of A (with self-loop): {loop_graph.in_degree('A')}\n")
    
    # Test 7: Testing removal operations
    print("7. Testing removal operations:")
    print(f"   Before removal: {simple_graph}")
    simple_graph.remove_edge('Y', 'Z')
    print(f"   After removing Y->Z: {simple_graph}")
    simple_graph.add_edge('Y', 'Z')
    print(f"   After adding back Y->Z: {simple_graph}\n")
    
    # Test 8: Testing predecessors and successors
    print("8. Testing predecessors and successors:")
    print(f"   Successors of X (outgoing): {simple_graph.get_neighbors('X')}")
    print(f"   Predecessors of X (incoming): {simple_graph.get_predecessors('X')}")
    print(f"   Successors of Z: {simple_graph.get_neighbors('Z')}")
    print(f"   Predecessors of Z: {simple_graph.get_predecessors('Z')}\n")
    
    # Test 9: Graph information summary
    print("9. Directed graph information summary:")
    info = simple_graph.get_graph_info()
    for key, value in info.items():
        if key != 'vertex_degrees':  # Skip detailed vertex info for summary
            print(f"   {key}: {value}")
    print()
    
    # Test 10: House of Nikolaus visualization
    print("10. Creating and visualizing the House of Nikolaus:")
    house_data = {
        'A': ['B', 'D'],    # Bottom left -> bottom right, top left
        'B': ['C'],         # Bottom right -> top right
        'C': ['E', 'A'],    # Top left -> top right, roof peak
        'D': ['C', 'B'],    # Top right -> bottom right, roof peak
        'E': ['D']          # Roof peak -> top left
    }
    house_graph = DirectedGraph(house_data)
    print(f"   {house_graph}")
    
    try:
        house_positions = {
            'A': (0, 0),    # Bottom left corner
            'B': (2, 0),    # Bottom right corner  
            'C': (0, 2),    # Top left corner
            'D': (2, 2),    # Top right corner
            'E': (1, 3)     # Roof peak
        }
        house_graph.visualize("House of Nikolaus (Directed)", 
                            figsize=(12, 10), 
                            positions=house_positions)
    except ImportError:
        print("   Matplotlib or NetworkX not available for visualization")
    
    print(f"\n   House of Nikolaus has {house_graph.vertex_count()} vertices and {house_graph.edge_count()} directed edges")
    
    print("\n=== DirectedGraph tests completed ===")


if __name__ == "__main__":
    main()