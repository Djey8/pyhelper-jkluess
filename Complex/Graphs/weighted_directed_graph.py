import networkx as nx
from typing import List, Set, Dict, Optional, Any, Union

import matplotlib.pyplot as plt


class WeightedDirectedGraph:
    """
    A weighted directed graph implementation using adjacency list representation.
    """
    
    def __init__(self, data: Optional[Dict[Any, List[tuple]]] = None):
        """
        Initialize a weighted directed graph.
        
        Args:
            data: Optional dictionary where keys are vertices and values are lists of tuples (neighbor, weight)
        """
        self._adjacency_list: Dict[Any, Dict[Any, Union[int, float]]] = {}
        
        if data:
            for vertex, neighbors in data.items():
                self.add_vertex(vertex)
                for neighbor_data in neighbors:
                    if isinstance(neighbor_data, tuple) and len(neighbor_data) == 2:
                        neighbor, weight = neighbor_data
                        self.add_edge(vertex, neighbor, weight)
                    else:
                        # Assume weight 1 if not provided as tuple
                        self.add_edge(vertex, neighbor_data, 1)
    
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
        
        # Remove all edges pointing to this vertex
        for v in self._adjacency_list:
            if vertex in self._adjacency_list[v]:
                del self._adjacency_list[v][vertex]
        
        # Remove the vertex itself
        del self._adjacency_list[vertex]
        return True
    
    def add_edge(self, from_vertex: Any, to_vertex: Any, weight: Union[int, float] = 1) -> bool:
        """
        Add a weighted directed edge from one vertex to another.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            weight: Weight of the edge (default: 1)
            
        Returns:
            bool: True if edge was added, False if edge already exists
        """
        if from_vertex not in self._adjacency_list:
            self.add_vertex(from_vertex)
        if to_vertex not in self._adjacency_list:
            self.add_vertex(to_vertex)
        
        if to_vertex not in self._adjacency_list[from_vertex]:
            self._adjacency_list[from_vertex][to_vertex] = weight
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
            
            del self._adjacency_list[from_vertex][to_vertex]
            return True
        return False
    
    def update_edge_weight(self, from_vertex: Any, to_vertex: Any, weight: Union[int, float]) -> bool:
        """
        Update the weight of an existing edge.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            weight: New weight for the edge
            
        Returns:
            bool: True if weight was updated, False if edge doesn't exist
        """
        if (from_vertex in self._adjacency_list and
            to_vertex in self._adjacency_list[from_vertex]):
            
            self._adjacency_list[from_vertex][to_vertex] = weight
            return True
        return False
    
    def get_edge_weight(self, from_vertex: Any, to_vertex: Any) -> Optional[Union[int, float]]:
        """
        Get the weight of an edge.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            
        Returns:
            Weight of the edge or None if edge doesn't exist
        """
        if (from_vertex in self._adjacency_list and
            to_vertex in self._adjacency_list[from_vertex]):
            return self._adjacency_list[from_vertex][to_vertex]
        return None
    
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
            return list(self._adjacency_list[vertex].keys())
        return []
    
    def get_weighted_neighbors(self, vertex: Any) -> List[tuple]:
        """
        Get all outgoing neighbors with their weights.
        
        Args:
            vertex: The vertex to get neighbors for
            
        Returns:
            List of tuples (neighbor, weight)
        """
        if vertex in self._adjacency_list:
            return [(neighbor, weight) for neighbor, weight in self._adjacency_list[vertex].items()]
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
    
    def get_weighted_predecessors(self, vertex: Any) -> List[tuple]:
        """
        Get all incoming neighbors (predecessors) with their weights.
        
        Args:
            vertex: The vertex to get predecessors for
            
        Returns:
            List of tuples (predecessor, weight)
        """
        if vertex not in self._adjacency_list:
            return []
        
        predecessors = []
        for v in self._adjacency_list:
            if vertex in self._adjacency_list[v]:
                weight = self._adjacency_list[v][vertex]
                predecessors.append((v, weight))
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
    
    def weighted_in_degree(self, vertex: Any) -> Union[int, float]:
        """
        Get the weighted in-degree of a vertex (sum of incoming edge weights).
        
        Args:
            vertex: The vertex to get weighted in-degree for
            
        Returns:
            Sum of weights of edges pointing to the vertex
        """
        return sum(weight for _, weight in self.get_weighted_predecessors(vertex))
    
    def weighted_out_degree(self, vertex: Any) -> Union[int, float]:
        """
        Get the weighted out-degree of a vertex (sum of outgoing edge weights).
        
        Args:
            vertex: The vertex to get weighted out-degree for
            
        Returns:
            Sum of weights of edges pointing from the vertex
        """
        if vertex not in self._adjacency_list:
            return 0
        return sum(self._adjacency_list[vertex].values())
    
    def get_edges(self) -> List[tuple]:
        """Get all weighted directed edges in the graph."""
        edges = []
        for vertex in self._adjacency_list:
            for neighbor, weight in self._adjacency_list[vertex].items():
                edges.append((vertex, neighbor, weight))
        return edges
    
    def vertex_count(self) -> int:
        """Get the number of vertices in the graph."""
        return len(self._adjacency_list)
    
    def edge_count(self) -> int:
        """Get the number of directed edges in the graph."""
        return sum(len(neighbors) for neighbors in self._adjacency_list.values())
    
    def total_weight(self) -> Union[int, float]:
        """Get the total weight of all edges in the graph."""
        return sum(weight for _, _, weight in self.get_edges())
    
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
    
    def visualize(self, title: str = "Weighted Directed Graph", figsize: tuple = (12, 9), positions: Optional[Dict[Any, tuple]] = None):
        """
        Visualize the weighted graph using matplotlib and networkx.
        
        Args:
            title: Title for the graph visualization
            figsize: Figure size as (width, height)
            positions: Optional dictionary mapping vertices to (x, y) coordinates
        """
        if not self._adjacency_list:
            print("Graph is empty - nothing to visualize")
            return
        
        # Create NetworkX directed graph
        G = nx.DiGraph()
        
        # Add vertices
        for vertex in self._adjacency_list:
            G.add_node(vertex)
        
        # Add weighted directed edges
        for from_vertex, to_vertex, weight in self.get_edges():
            G.add_edge(from_vertex, to_vertex, weight=weight)
        
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
                node_size=800,
                font_size=12,
                font_weight='bold',
                edge_color='gray',
                width=2,
                arrows=True,
                arrowsize=20,
                arrowstyle='->')
        
        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, font_color='red')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def degree(self, vertex: Any) -> int:
        """
        Get the total degree of a vertex (in-degree + out-degree).
        
        Args:
            vertex: The vertex to get the degree for
            
        Returns:
            int: The total degree of the vertex, or 0 if vertex doesn't exist
        """
        if vertex not in self._adjacency_list:
            return 0
        return self.in_degree(vertex) + self.out_degree(vertex)
    
    def weighted_degree(self, vertex: Any) -> Union[int, float]:
        """
        Get the total weighted degree of a vertex (weighted in-degree + weighted out-degree).
        
        Args:
            vertex: The vertex to get the weighted degree for
            
        Returns:
            The total weighted degree of the vertex, or 0 if vertex doesn't exist
        """
        if vertex not in self._adjacency_list:
            return 0
        return self.weighted_in_degree(vertex) + self.weighted_out_degree(vertex)

    def get_degree_sequence(self) -> Dict[str, List[int]]:
        """Get the degree sequences of the graph (in-degrees, out-degrees, and total degrees)."""
        in_degrees = [self.in_degree(vertex) for vertex in self._adjacency_list]
        out_degrees = [self.out_degree(vertex) for vertex in self._adjacency_list]
        total_degrees = [self.degree(vertex) for vertex in self._adjacency_list]
        
        return {
            'in_degrees': sorted(in_degrees, reverse=True),
            'out_degrees': sorted(out_degrees, reverse=True),
            'total_degrees': sorted(total_degrees, reverse=True)
        }
    
    def get_weighted_degree_sequence(self) -> Dict[str, List[Union[int, float]]]:
        """Get the weighted degree sequences of the graph."""
        weighted_in_degrees = [self.weighted_in_degree(vertex) for vertex in self._adjacency_list]
        weighted_out_degrees = [self.weighted_out_degree(vertex) for vertex in self._adjacency_list]
        weighted_total_degrees = [self.weighted_degree(vertex) for vertex in self._adjacency_list]
        
        return {
            'weighted_in_degrees': sorted(weighted_in_degrees, reverse=True),
            'weighted_out_degrees': sorted(weighted_out_degrees, reverse=True),
            'weighted_total_degrees': sorted(weighted_total_degrees, reverse=True)
        }

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

    def get_graph_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the weighted directed graph structure."""
        if not self._adjacency_list:
            return {
                "vertices": 0,
                "edges": 0,
                "total_weight": 0,
                "is_simple": True,
                "degree_sequences": {'in_degrees': [], 'out_degrees': [], 'total_degrees': []},
                "weighted_degree_sequences": {'weighted_in_degrees': [], 'weighted_out_degrees': [], 'weighted_total_degrees': []},
                "weight_statistics": {'min_weight': 0, 'max_weight': 0, 'average_weight': 0, 'total_weight': 0},
                "vertex_degrees": {}
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
        
        return {
            "vertices": self.vertex_count(),
            "edges": self.edge_count(),
            "total_weight": self.total_weight(),
            "is_simple": self.is_simple_graph(),
            "degree_sequences": self.get_degree_sequence(),
            "weighted_degree_sequences": self.get_weighted_degree_sequence(),
            "weight_statistics": self.get_weight_statistics(),
            "vertex_degrees": vertex_degrees
        }

    def print_graph_analysis(self):
        """Print a detailed analysis of the weighted directed graph."""
        info = self.get_graph_info()
        
        print("=== Weighted Directed Graph Theory Analysis ===")
        print(f"Weighted Directed Graph G = (V, E, w) with |V| = {info['vertices']} vertices and |E| = {info['edges']} weighted edges")
        print(f"Total weight of all edges: {info['total_weight']}")
        print()
        
        print("Basic Properties:")
        print(f"  • Simple graph (no self-loops): {'Yes' if info['is_simple'] else 'No'}")
        print()
        
        print("Weight Statistics:")
        ws = info['weight_statistics']
        print(f"  • Minimum edge weight: {ws['min_weight']}")
        print(f"  • Maximum edge weight: {ws['max_weight']}")
        print(f"  • Average edge weight: {ws['average_weight']:.2f}")
        print(f"  • Total weight: {ws['total_weight']}")
        print()
        
        print("Degree Sequences (Unweighted):")
        ds = info['degree_sequences']
        print(f"  • In-degree sequence: {ds['in_degrees']}")
        print(f"  • Out-degree sequence: {ds['out_degrees']}")
        print(f"  • Total degree sequence: {ds['total_degrees']}")
        print()
        
        print("Weighted Degree Sequences:")
        wds = info['weighted_degree_sequences']
        print(f"  • Weighted in-degree sequence: {wds['weighted_in_degrees']}")
        print(f"  • Weighted out-degree sequence: {wds['weighted_out_degrees']}")
        print(f"  • Weighted total degree sequence: {wds['weighted_total_degrees']}")
        print()
        
        print("Individual Vertex Analysis:")
        for vertex in sorted(info['vertex_degrees'].keys()):
            degrees = info['vertex_degrees'][vertex]
            weighted_successors = sorted(self.get_weighted_neighbors(vertex))
            weighted_predecessors = sorted(self.get_weighted_predecessors(vertex))
            print(f"  Vertex {vertex}:")
            print(f"    • deg⁺({vertex}) = {degrees['out_degree']}, weighted deg⁺({vertex}) = {degrees['weighted_out_degree']}")
            print(f"      Successors: {weighted_successors}")
            print(f"    • deg⁻({vertex}) = {degrees['in_degree']}, weighted deg⁻({vertex}) = {degrees['weighted_in_degree']}")
            print(f"      Predecessors: {weighted_predecessors}")
            print(f"    • Total: deg({vertex}) = {degrees['total_degree']}, weighted deg({vertex}) = {degrees['weighted_total_degree']}")
        print()
        
        print("Weighted Graph Theory Concepts:")
        print("  • Edge weight: w(u,v) = weight of directed edge from u to v")
        print("  • Weighted out-degree: w⁺(v) = Σ w(v,u) for all edges (v,u)")
        print("  • Weighted in-degree: w⁻(v) = Σ w(u,v) for all edges (u,v)")
        print("  • Weighted total degree: w(v) = w⁺(v) + w⁻(v)")

    # Path and reachability methods
    def find_path(self, start: Any, end: Any) -> Optional[List[Any]]:
        """
        Find a path from start vertex to end vertex using BFS (unweighted shortest path).
        
        Theory: A path (Weg/Pfad) from v to v' is a sequence of vertices v₀, v₁, ..., vₖ where:
        - v₀ = v (start vertex)
        - vₖ = v' (end vertex)
        - (vᵢ, vᵢ₊₁) ∈ E for i = 0, ..., k-1 (directed edges)
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
            for neighbor in self._adjacency_list[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        if len(visited) != len(self._adjacency_list):
            return False
        
        # Reverse DFS: check if all vertices can reach first vertex
        # Create reverse graph
        reverse_adj = {v: {} for v in self._adjacency_list}
        for vertex in self._adjacency_list:
            for neighbor, weight in self._adjacency_list[vertex].items():
                reverse_adj[neighbor][vertex] = weight
        
        visited = set()
        stack = [first_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            for neighbor in reverse_adj[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
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
        reverse_adj = {v: {} for v in self._adjacency_list}
        for vertex in self._adjacency_list:
            for neighbor in self._adjacency_list[vertex]:
                reverse_adj[neighbor][vertex] = self._adjacency_list[vertex][neighbor]
        
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
    def get_adjacency_matrix(self) -> List[List[Union[int, float]]]:
        """
        Get the weighted adjacency matrix representation of the directed graph.
        
        Theory: For weighted directed graph G = (V, E) with V = {v₁, ..., vₙ}, the weighted
        adjacency matrix A ∈ ℝⁿˣⁿ is defined as:
        - aᵢⱼ = weight if directed edge from vertex vᵢ to vⱼ exists
        - aᵢⱼ = 0 if no directed edge from vertex vᵢ to vⱼ exists
        
        For directed graphs, the matrix is generally not symmetric.
        
        Returns:
            n×n matrix where matrix[i][j] = edge weight if edge (i,j) exists, 0 otherwise
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
    def from_adjacency_matrix(cls, matrix: List[List[Union[int, float]]], vertices: Optional[List[Any]] = None) -> 'WeightedDirectedGraph':
        """
        Create a weighted directed graph from an adjacency matrix.
        
        Theory: Converts a weighted adjacency matrix representation back to a directed graph structure.
        
        Args:
            matrix: n×n adjacency matrix where matrix[i][j] contains the edge weight (0 = no edge)
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
        
        # Add edges from matrix
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    graph.add_edge(vertices[i], vertices[j], matrix[i][j])
        
        return graph

    def __str__(self) -> str:
        """String representation of the weighted graph."""
        if not self._adjacency_list:
            return "Empty weighted graph"
        
        result = "Weighted Directed Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            neighbors = sorted([(neighbor, weight) for neighbor, weight in self._adjacency_list[vertex].items()])
            result += f"  {vertex} -> {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the weighted graph."""
        return f"WeightedDirectedGraph(vertices={self.vertex_count()}, edges={self.edge_count()}, total_weight={self.total_weight()})"


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


if __name__ == "__main__":
    main()