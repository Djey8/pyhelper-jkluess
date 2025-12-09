import networkx as nx
from typing import List, Set, Dict, Optional, Any, Union
import matplotlib.pyplot as plt


class Graph:
    """
    A unified graph implementation that supports all graph types:
    - Undirected graphs (directed=False, weighted=False)
    - Directed graphs (directed=True, weighted=False)
    - Weighted undirected graphs (directed=False, weighted=True)
    - Weighted directed graphs (directed=True, weighted=True)
    
    The behavior adapts based on initialization parameters.
    """
    
    def __init__(self, directed: bool = False, weighted: bool = False, data: Optional[Dict[Any, List]] = None):
        """
        Initialize a graph with specified type.
        
        Args:
            directed: If True, creates a directed graph; if False, undirected
            weighted: If True, edges have weights; if False, unweighted
            data: Optional dictionary where keys are vertices and values are:
                  - For unweighted: lists of adjacent vertices
                  - For weighted: lists of tuples (neighbor, weight)
        """
        self._directed = directed
        self._weighted = weighted
        
        # For weighted graphs: Dict[vertex, Dict[neighbor, weight]]
        # For unweighted graphs: Dict[vertex, Set[neighbor]]
        if weighted:
            self._adjacency_list: Dict[Any, Dict[Any, Union[int, float]]] = {}
        else:
            self._adjacency_list: Dict[Any, Set[Any]] = {}
        
        if data:
            for vertex, neighbors in data.items():
                self.add_vertex(vertex)
                if weighted:
                    for neighbor_data in neighbors:
                        if isinstance(neighbor_data, tuple) and len(neighbor_data) == 2:
                            neighbor, weight = neighbor_data
                            self.add_edge(vertex, neighbor, weight)
                        else:
                            # Assume weight 1 if not provided as tuple
                            self.add_edge(vertex, neighbor_data, 1)
                else:
                    for neighbor in neighbors:
                        self.add_edge(vertex, neighbor)
    
    @property
    def is_directed(self) -> bool:
        """Check if graph is directed."""
        return self._directed
    
    @property
    def is_weighted(self) -> bool:
        """Check if graph is weighted."""
        return self._weighted
    
    def add_vertex(self, vertex: Any) -> bool:
        """
        Add a vertex to the graph.
        
        Args:
            vertex: The vertex to add
            
        Returns:
            bool: True if vertex was added, False if it already exists
        """
        if vertex not in self._adjacency_list:
            if self._weighted:
                self._adjacency_list[vertex] = {}
            else:
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
        
        # Remove all edges going out from this vertex (clear its adjacency list)
        self._adjacency_list[vertex].clear()
        
        # Remove all edges coming into this vertex from other vertices
        if self._weighted:
            for v in self._adjacency_list:
                if vertex in self._adjacency_list[v]:
                    del self._adjacency_list[v][vertex]
        else:
            for v in self._adjacency_list:
                self._adjacency_list[v].discard(vertex)
        
        # Remove the vertex itself
        del self._adjacency_list[vertex]
        return True
    
    def add_edge(self, vertex1: Any, vertex2: Any, weight: Union[int, float] = 1) -> bool:
        """
        Add an edge between two vertices.
        
        Args:
            vertex1: First vertex (source for directed graphs)
            vertex2: Second vertex (destination for directed graphs)
            weight: Weight of the edge (only used if graph is weighted)
            
        Returns:
            bool: True if edge was added, False if edge already exists
        """
        if vertex1 not in self._adjacency_list:
            self.add_vertex(vertex1)
        if vertex2 not in self._adjacency_list:
            self.add_vertex(vertex2)
        
        if self._weighted:
            # Weighted graph
            if vertex2 not in self._adjacency_list[vertex1]:
                self._adjacency_list[vertex1][vertex2] = weight
                if not self._directed:
                    self._adjacency_list[vertex2][vertex1] = weight
                return True
        else:
            # Unweighted graph
            if vertex2 not in self._adjacency_list[vertex1]:
                self._adjacency_list[vertex1].add(vertex2)
                if not self._directed:
                    self._adjacency_list[vertex2].add(vertex1)
                return True
        return False
    
    def remove_edge(self, vertex1: Any, vertex2: Any) -> bool:
        """
        Remove an edge between two vertices.
        
        Args:
            vertex1: First vertex (source for directed graphs)
            vertex2: Second vertex (destination for directed graphs)
            
        Returns:
            bool: True if edge was removed, False if edge doesn't exist
        """
        if vertex1 not in self._adjacency_list:
            return False
        
        if self._weighted:
            if vertex2 in self._adjacency_list[vertex1]:
                del self._adjacency_list[vertex1][vertex2]
                if not self._directed and vertex1 in self._adjacency_list[vertex2]:
                    del self._adjacency_list[vertex2][vertex1]
                return True
        else:
            if vertex2 in self._adjacency_list[vertex1]:
                self._adjacency_list[vertex1].discard(vertex2)
                if not self._directed:
                    self._adjacency_list[vertex2].discard(vertex1)
                return True
        return False
    
    def update_edge_weight(self, vertex1: Any, vertex2: Any, weight: Union[int, float]) -> bool:
        """
        Update the weight of an existing edge (only for weighted graphs).
        
        Args:
            vertex1: First vertex (source for directed graphs)
            vertex2: Second vertex (destination for directed graphs)
            weight: New weight for the edge
            
        Returns:
            bool: True if weight was updated, False if edge doesn't exist or graph is unweighted
        """
        if not self._weighted:
            return False
        
        if vertex1 in self._adjacency_list and vertex2 in self._adjacency_list[vertex1]:
            self._adjacency_list[vertex1][vertex2] = weight
            if not self._directed and vertex2 in self._adjacency_list:
                self._adjacency_list[vertex2][vertex1] = weight
            return True
        return False
    
    def get_edge_weight(self, vertex1: Any, vertex2: Any) -> Optional[Union[int, float]]:
        """
        Get the weight of an edge (only for weighted graphs).
        
        Args:
            vertex1: First vertex (source for directed graphs)
            vertex2: Second vertex (destination for directed graphs)
            
        Returns:
            Weight of the edge or None if edge doesn't exist or graph is unweighted
        """
        if not self._weighted:
            return None
        
        if vertex1 in self._adjacency_list and vertex2 in self._adjacency_list[vertex1]:
            return self._adjacency_list[vertex1][vertex2]
        return None
    
    def has_vertex(self, vertex: Any) -> bool:
        """Check if a vertex exists in the graph."""
        return vertex in self._adjacency_list
    
    def has_edge(self, vertex1: Any, vertex2: Any) -> bool:
        """Check if an edge exists between two vertices."""
        if vertex1 not in self._adjacency_list:
            return False
        return vertex2 in self._adjacency_list[vertex1]
    
    def get_vertices(self) -> List[Any]:
        """Get all vertices in the graph."""
        return list(self._adjacency_list.keys())
    
    def get_neighbors(self, vertex: Any) -> List[Any]:
        """
        Get all neighbors of a vertex.
        For directed graphs, returns outgoing neighbors (successors).
        """
        if vertex not in self._adjacency_list:
            return []
        
        if self._weighted:
            return list(self._adjacency_list[vertex].keys())
        else:
            return list(self._adjacency_list[vertex])
    
    def get_weighted_neighbors(self, vertex: Any) -> List[tuple]:
        """
        Get all neighbors with their weights (only for weighted graphs).
        
        Args:
            vertex: The vertex to get neighbors for
            
        Returns:
            List of tuples (neighbor, weight) or empty list if unweighted
        """
        if not self._weighted or vertex not in self._adjacency_list:
            return []
        
        return [(neighbor, weight) for neighbor, weight in self._adjacency_list[vertex].items()]
    
    def get_predecessors(self, vertex: Any) -> List[Any]:
        """
        Get all incoming neighbors (predecessors) of a vertex.
        Only relevant for directed graphs. For undirected graphs, same as get_neighbors.
        
        Args:
            vertex: The vertex to get predecessors for
            
        Returns:
            List of vertices that have edges pointing to the given vertex
        """
        if vertex not in self._adjacency_list:
            return []
        
        if not self._directed:
            return self.get_neighbors(vertex)
        
        predecessors = []
        for v in self._adjacency_list:
            if vertex in self._adjacency_list[v]:
                predecessors.append(v)
        return predecessors
    
    def get_weighted_predecessors(self, vertex: Any) -> List[tuple]:
        """
        Get all incoming neighbors (predecessors) with their weights.
        Only relevant for weighted directed graphs.
        
        Args:
            vertex: The vertex to get predecessors for
            
        Returns:
            List of tuples (predecessor, weight)
        """
        if not self._weighted or vertex not in self._adjacency_list:
            return []
        
        if not self._directed:
            return self.get_weighted_neighbors(vertex)
        
        predecessors = []
        for v in self._adjacency_list:
            if vertex in self._adjacency_list[v]:
                weight = self._adjacency_list[v][vertex]
                predecessors.append((v, weight))
        return predecessors
    
    def degree(self, vertex: Any) -> int:
        """
        Get the degree of a vertex.
        For undirected graphs: number of adjacent vertices
        For directed graphs: in-degree + out-degree
        """
        if vertex not in self._adjacency_list:
            return 0
        
        if not self._directed:
            return len(self._adjacency_list[vertex])
        else:
            return self.in_degree(vertex) + self.out_degree(vertex)
    
    def in_degree(self, vertex: Any) -> int:
        """
        Get the in-degree of a vertex (number of incoming edges).
        Only relevant for directed graphs. For undirected graphs, same as degree.
        """
        if not self._directed:
            return self.degree(vertex)
        return len(self.get_predecessors(vertex))
    
    def out_degree(self, vertex: Any) -> int:
        """
        Get the out-degree of a vertex (number of outgoing edges).
        Only relevant for directed graphs. For undirected graphs, same as degree.
        """
        if vertex not in self._adjacency_list:
            return 0
        
        if not self._directed:
            return self.degree(vertex)
        return len(self._adjacency_list[vertex])
    
    def weighted_degree(self, vertex: Any) -> Union[int, float]:
        """
        Get the weighted degree of a vertex (sum of edge weights).
        Only for weighted graphs.
        """
        if not self._weighted or vertex not in self._adjacency_list:
            return 0
        
        if not self._directed:
            return sum(self._adjacency_list[vertex].values())
        else:
            return self.weighted_in_degree(vertex) + self.weighted_out_degree(vertex)
    
    def weighted_in_degree(self, vertex: Any) -> Union[int, float]:
        """
        Get the weighted in-degree of a vertex (sum of incoming edge weights).
        Only for weighted directed graphs.
        """
        if not self._weighted:
            return 0
        
        if not self._directed:
            return self.weighted_degree(vertex)
        
        return sum(weight for _, weight in self.get_weighted_predecessors(vertex))
    
    def weighted_out_degree(self, vertex: Any) -> Union[int, float]:
        """
        Get the weighted out-degree of a vertex (sum of outgoing edge weights).
        Only for weighted directed graphs.
        """
        if not self._weighted or vertex not in self._adjacency_list:
            return 0
        
        if not self._directed:
            return self.weighted_degree(vertex)
        
        return sum(self._adjacency_list[vertex].values())
    
    def get_edges(self) -> List[tuple]:
        """
        Get all edges in the graph.
        Returns:
            - For unweighted: list of tuples (vertex1, vertex2)
            - For weighted: list of tuples (vertex1, vertex2, weight)
        """
        edges = []
        seen = set()
        
        for vertex in self._adjacency_list:
            if self._weighted:
                for neighbor, weight in self._adjacency_list[vertex].items():
                    if not self._directed:
                        edge_key = tuple(sorted([vertex, neighbor]))
                        if edge_key not in seen:
                            edges.append((vertex, neighbor, weight))
                            seen.add(edge_key)
                    else:
                        edges.append((vertex, neighbor, weight))
            else:
                for neighbor in self._adjacency_list[vertex]:
                    if not self._directed:
                        edge_key = tuple(sorted([vertex, neighbor]))
                        if edge_key not in seen:
                            edges.append((vertex, neighbor))
                            seen.add(edge_key)
                    else:
                        edges.append((vertex, neighbor))
        
        return edges
    
    def vertex_count(self) -> int:
        """Get the number of vertices in the graph."""
        return len(self._adjacency_list)
    
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return len(self.get_edges())
    
    def total_weight(self) -> Union[int, float]:
        """Get the total weight of all edges (only for weighted graphs)."""
        if not self._weighted:
            return 0
        
        edges = self.get_edges()
        return sum(weight for *_, weight in edges)
    
    def is_simple_graph(self) -> bool:
        """Check if the graph is simple (no self-loops)."""
        for vertex in self._adjacency_list:
            if vertex in self._adjacency_list[vertex]:
                return False
        return True
    
    # Path and reachability methods
    def find_path(self, start: Any, end: Any) -> Optional[List[Any]]:
        """
        Find a path from start vertex to end vertex using BFS.
        For directed graphs: follows edge directions.
        For unweighted graphs: finds shortest path by edge count.
        For weighted graphs: finds a path (use dijkstra for shortest weighted path).
        """
        if start not in self._adjacency_list or end not in self._adjacency_list:
            return None
        
        if start == end:
            return [start]
        
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            vertex, path = queue.pop(0)
            
            neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
            
            for neighbor in neighbors:
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def is_reachable(self, start: Any, end: Any) -> bool:
        """Check if end vertex is reachable from start vertex."""
        return self.find_path(start, end) is not None
    
    def path_length(self, path: List[Any]) -> int:
        """Calculate the length of a path (number of edges)."""
        if not path or len(path) < 2:
            return 0
        return len(path) - 1
    
    def path_weight(self, path: List[Any]) -> Union[int, float]:
        """
        Calculate the total weight of a path (only for weighted graphs).
        For unweighted graphs, returns the edge count.
        """
        if not path or len(path) < 2:
            return 0
        
        if not self._weighted:
            return self.path_length(path)
        
        total = 0
        for i in range(len(path) - 1):
            if path[i] not in self._adjacency_list or path[i+1] not in self._adjacency_list[path[i]]:
                return 0  # Invalid path
            total += self._adjacency_list[path[i]][path[i+1]]
        
        return total
    
    def dfs(self, start: Any, end: Optional[Any] = None, visited: Optional[Set[Any]] = None) -> List[Any]:
        """
        Perform Depth-First Search (DFS) starting from a given vertex.
        
        Args:
            start: Starting vertex for the search
            end: Optional ending vertex. If provided, stops when reached.
            visited: Optional set of already visited vertices (for internal use)
            
        Returns:
            List of vertices in the order they were visited during DFS
            
        Example:
            >>> g = Graph()
            >>> g.add_edge('A', 'B')
            >>> g.add_edge('B', 'C')
            >>> g.add_edge('A', 'D')
            >>> traversal = g.dfs('A')
            >>> print(traversal)  # e.g., ['A', 'B', 'C', 'D']
        """
        if start not in self._adjacency_list:
            return []
        
        if visited is None:
            visited = set()
        
        result = []
        
        def dfs_recursive(vertex: Any):
            if vertex in visited:
                return
            
            visited.add(vertex)
            result.append(vertex)
            
            if end and vertex == end:
                return
            
            neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
            for neighbor in sorted(neighbors):  # Sort for consistent ordering
                if neighbor not in visited:
                    dfs_recursive(neighbor)
        
        dfs_recursive(start)
        return result
    
    def iter_dfs(self, start: Any, end: Optional[Any] = None, visited: Optional[Set[Any]] = None):
        """
        Generator for Depth-First Search (DFS).
        Yields vertices one at a time (memory efficient).
        
        Args:
            start: Starting vertex for the search
            end: Optional ending vertex. If provided, stops when reached.
            visited: Set of already visited vertices (for internal use)
            
        Yields:
            Vertices in DFS order
            
        Example:
            >>> for vertex in graph.iter_dfs('A'):
            ...     print(vertex)
        """
        if start not in self._adjacency_list:
            return
        
        if visited is None:
            visited = set()
        
        visited.add(start)
        yield start
        
        if end and start == end:
            return
        
        neighbors = list(self._adjacency_list[start].keys()) if self._weighted else list(self._adjacency_list[start])
        for neighbor in sorted(neighbors):
            if neighbor not in visited:
                yield from self.iter_dfs(neighbor, end, visited)
    
    def bfs(self, start: Any, end: Optional[Any] = None) -> List[Any]:
        """
        Perform Breadth-First Search (BFS) starting from a given vertex.
        
        Args:
            start: Starting vertex for the search
            end: Optional ending vertex. If provided, stops when reached.
            
        Returns:
            List of vertices in the order they were visited during BFS
            
        Example:
            >>> g = Graph()
            >>> g.add_edge('A', 'B')
            >>> g.add_edge('A', 'C')
            >>> g.add_edge('B', 'D')
            >>> traversal = g.bfs('A')
            >>> print(traversal)  # ['A', 'B', 'C', 'D']
        """
        if start not in self._adjacency_list:
            return []
        
        queue = [start]
        visited = {start}
        result = []
        
        while queue:
            vertex = queue.pop(0)
            result.append(vertex)
            
            if end and vertex == end:
                break
            
            neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
            for neighbor in sorted(neighbors):  # Sort for consistent ordering
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def iter_bfs(self, start: Any, end: Optional[Any] = None):
        """
        Generator for Breadth-First Search (BFS).
        Yields vertices one at a time (memory efficient).
        
        Args:
            start: Starting vertex for the search
            end: Optional ending vertex. If provided, stops when reached.
            
        Yields:
            Vertices in BFS order
            
        Example:
            >>> for vertex in graph.iter_bfs('A'):
            ...     print(vertex)
        """
        if start not in self._adjacency_list:
            return
        
        queue = [start]
        visited = {start}
        
        while queue:
            vertex = queue.pop(0)
            yield vertex
            
            if end and vertex == end:
                return
            
            neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
            for neighbor in sorted(neighbors):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    def find_shortest_path(self, start: Any, end: Any) -> Optional[tuple]:
        """
        Find the shortest path between two vertices.
        
        For unweighted graphs: Uses BFS, returns shortest path by edge count.
        For weighted graphs: Uses Dijkstra's algorithm, returns shortest path by total weight.
        Works for both directed and undirected graphs.
        
        Args:
            start: Starting vertex
            end: Ending vertex
            
        Returns:
            Tuple (path, distance) where:
                - path is a list of vertices representing the shortest path
                - distance is the total weight (or edge count for unweighted)
            Returns None if no path exists.
            
        Example:
            >>> g = Graph(weighted=True)
            >>> g.add_edge('A', 'B', 4)
            >>> g.add_edge('A', 'C', 2)
            >>> g.add_edge('C', 'B', 1)
            >>> path, dist = g.find_shortest_path('A', 'B')
            >>> print(path)  # ['A', 'C', 'B']
            >>> print(dist)  # 3
        """
        if start not in self._adjacency_list or end not in self._adjacency_list:
            return None
        
        if start == end:
            return ([start], 0)
        
        if not self._weighted:
            # BFS for unweighted graphs (shortest by edge count)
            queue = [(start, [start], 0)]
            visited = {start}
            
            while queue:
                vertex, path, dist = queue.pop(0)
                
                neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
                
                for neighbor in neighbors:
                    if neighbor == end:
                        return (path + [neighbor], dist + 1)
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor], dist + 1))
            
            return None
        else:
            # Dijkstra's algorithm for weighted graphs
            import heapq
            
            # Initialize distances and previous vertices
            distances = {vertex: float('inf') for vertex in self._adjacency_list}
            distances[start] = 0
            previous = {vertex: None for vertex in self._adjacency_list}
            
            # Priority queue: (distance, vertex)
            pq = [(0, start)]
            visited = set()
            
            while pq:
                current_dist, current = heapq.heappop(pq)
                
                if current in visited:
                    continue
                
                visited.add(current)
                
                if current == end:
                    # Reconstruct path
                    path = []
                    node = end
                    while node is not None:
                        path.insert(0, node)
                        node = previous[node]
                    return (path, distances[end])
                
                # Check all neighbors
                for neighbor, weight in self._adjacency_list[current].items():
                    if neighbor not in visited:
                        new_dist = current_dist + weight
                        if new_dist < distances[neighbor]:
                            distances[neighbor] = new_dist
                            previous[neighbor] = current
                            heapq.heappush(pq, (new_dist, neighbor))
            
            return None
    
    def dijkstra(self, start: Any) -> Dict[Any, tuple]:
        """
        Run Dijkstra's algorithm to find shortest paths from start vertex to all other vertices.
        Only works for weighted graphs.
        
        Args:
            start: Starting vertex
            
        Returns:
            Dictionary mapping each vertex to (distance, path) from start vertex.
            Returns empty dict if graph is unweighted or start vertex doesn't exist.
            
        Example:
            >>> g = Graph(weighted=True)
            >>> g.add_edge('A', 'B', 4)
            >>> g.add_edge('A', 'C', 2)
            >>> g.add_edge('C', 'B', 1)
            >>> result = g.dijkstra('A')
            >>> print(result['B'])  # (3, ['A', 'C', 'B'])
        """
        if not self._weighted or start not in self._adjacency_list:
            return {}
        
        import heapq
        
        # Initialize distances and previous vertices
        distances = {vertex: float('inf') for vertex in self._adjacency_list}
        distances[start] = 0
        previous = {vertex: None for vertex in self._adjacency_list}
        
        # Priority queue: (distance, vertex)
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check all neighbors
            for neighbor, weight in self._adjacency_list[current].items():
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct paths
        result = {}
        for vertex in self._adjacency_list:
            if distances[vertex] != float('inf'):
                path = []
                node = vertex
                while node is not None:
                    path.insert(0, node)
                    node = previous[node]
                result[vertex] = (distances[vertex], path)
            else:
                result[vertex] = (float('inf'), [])
        
        return result
    
    def minimum_spanning_tree_kruskal(self) -> Optional['Graph']:
        """
        Find the Minimum Spanning Tree (MST) using Kruskal's algorithm.
        Only works for undirected weighted graphs.
        
        Returns:
            New Graph object representing the MST, or None if graph is directed or unweighted
            
        Example:
            >>> g = Graph(directed=False, weighted=True)
            >>> g.add_edge('A', 'B', 4)
            >>> g.add_edge('A', 'C', 2)
            >>> g.add_edge('B', 'C', 1)
            >>> g.add_edge('B', 'D', 5)
            >>> mst = g.minimum_spanning_tree_kruskal()
            >>> print(mst.get_edges())  # [('B', 'C', 1), ('A', 'C', 2), ('B', 'D', 5)]
        """
        if self._directed or not self._weighted:
            return None
        
        if not self._adjacency_list:
            return Graph(directed=False, weighted=True)
        
        # Union-Find (Disjoint Set) data structure
        class UnionFind:
            def __init__(self, vertices):
                self.parent = {v: v for v in vertices}
                self.rank = {v: 0 for v in vertices}
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])  # Path compression
                return self.parent[x]
            
            def union(self, x, y):
                root_x, root_y = self.find(x), self.find(y)
                if root_x != root_y:
                    # Union by rank
                    if self.rank[root_x] < self.rank[root_y]:
                        self.parent[root_x] = root_y
                    elif self.rank[root_x] > self.rank[root_y]:
                        self.parent[root_y] = root_x
                    else:
                        self.parent[root_y] = root_x
                        self.rank[root_x] += 1
                    return True
                return False
        
        # Get all edges and sort by weight
        edges = self.get_edges()
        edges.sort(key=lambda x: x[2])  # Sort by weight
        
        # Initialize Union-Find and MST
        uf = UnionFind(self.get_vertices())
        mst = Graph(directed=False, weighted=True)
        
        # Add all vertices to MST
        for vertex in self.get_vertices():
            mst.add_vertex(vertex)
        
        # Kruskal's algorithm
        for u, v, weight in edges:
            if uf.union(u, v):  # If adding this edge doesn't create a cycle
                mst.add_edge(u, v, weight)
                # Stop when we have n-1 edges (complete MST)
                if mst.edge_count() == self.vertex_count() - 1:
                    break
        
        return mst
    
    def minimum_spanning_tree_prim(self) -> Optional['Graph']:
        """
        Find the Minimum Spanning Tree (MST) using Prim's algorithm.
        Only works for undirected weighted graphs.
        
        Returns:
            New Graph object representing the MST, or None if graph is directed or unweighted
            
        Example:
            >>> g = Graph(directed=False, weighted=True)
            >>> g.add_edge('A', 'B', 4)
            >>> g.add_edge('A', 'C', 2)
            >>> g.add_edge('B', 'C', 1)
            >>> g.add_edge('B', 'D', 5)
            >>> mst = g.minimum_spanning_tree_prim()
            >>> print(mst.total_weight())  # 8
        """
        if self._directed or not self._weighted:
            return None
        
        if not self._adjacency_list:
            return Graph(directed=False, weighted=True)
        
        import heapq
        
        # Start with arbitrary vertex
        start_vertex = next(iter(self._adjacency_list))
        mst = Graph(directed=False, weighted=True)
        
        # Add all vertices to MST
        for vertex in self.get_vertices():
            mst.add_vertex(vertex)
        
        # Keep track of vertices in MST
        in_mst = {start_vertex}
        
        # Priority queue of edges: (weight, u, v)
        pq = []
        
        # Add all edges from start vertex
        for neighbor, weight in self._adjacency_list[start_vertex].items():
            heapq.heappush(pq, (weight, start_vertex, neighbor))
        
        while pq and len(in_mst) < self.vertex_count():
            weight, u, v = heapq.heappop(pq)
            
            # Skip if both vertices are already in MST
            if v in in_mst:
                continue
            
            # Add edge to MST
            mst.add_edge(u, v, weight)
            in_mst.add(v)
            
            # Add all edges from newly added vertex
            for neighbor, edge_weight in self._adjacency_list[v].items():
                if neighbor not in in_mst:
                    heapq.heappush(pq, (edge_weight, v, neighbor))
        
        return mst
    
    def is_simple_path(self, path: List[Any]) -> bool:
        """Check if a path is simple (all vertices are pairwise distinct)."""
        if not path:
            return True
        return len(path) == len(set(path))
    
    # Cycle detection methods
    def has_cycle(self) -> bool:
        """Check if the graph contains any cycle."""
        if not self._adjacency_list:
            return False
        
        if self._directed:
            # For directed graphs: use DFS with 3 colors
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {vertex: WHITE for vertex in self._adjacency_list}
            
            def dfs_has_cycle(vertex: Any) -> bool:
                color[vertex] = GRAY
                
                neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
                for neighbor in neighbors:
                    if color[neighbor] == GRAY:  # Back edge
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
        else:
            # For undirected graphs: use DFS with parent tracking
            visited = set()
            
            def dfs_undirected(vertex: Any, parent: Optional[Any]) -> bool:
                visited.add(vertex)
                
                neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        if dfs_undirected(neighbor, vertex):
                            return True
                    elif neighbor != parent:
                        return True
                return False
            
            for vertex in self._adjacency_list:
                if vertex not in visited:
                    if dfs_undirected(vertex, None):
                        return True
            return False
    
    def find_all_cycles(self) -> List[List[Any]]:
        """
        Find all simple cycles in the graph.
        
        Returns:
            List of cycles, where each cycle is represented as a list of vertices
        """
        cycles = []
        
        if not self._adjacency_list:
            return cycles
        
        if self._directed:
            # For directed graphs: Find all cycles using Johnson's algorithm approach
            visited_global = set()
            
            def find_cycles_from_vertex(start_vertex: Any):
                stack = [start_vertex]
                path = [start_vertex]
                path_set = {start_vertex}
                blocked = set()
                B = {v: set() for v in self._adjacency_list}
                
                def unblock(vertex: Any):
                    blocked.discard(vertex)
                    for w in B[vertex]:
                        if w in blocked:
                            unblock(w)
                    B[vertex].clear()
                
                def dfs(vertex: Any) -> bool:
                    found_cycle = False
                    blocked.add(vertex)
                    
                    neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
                    for neighbor in neighbors:
                        if neighbor == start_vertex:
                            # Found cycle back to start
                            cycles.append(path + [start_vertex])
                            found_cycle = True
                        elif neighbor not in blocked and neighbor not in visited_global:
                            path.append(neighbor)
                            path_set.add(neighbor)
                            if dfs(neighbor):
                                found_cycle = True
                            path.pop()
                            path_set.discard(neighbor)
                    
                    if found_cycle:
                        unblock(vertex)
                    else:
                        for neighbor in (list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])):
                            B[neighbor].add(vertex)
                    
                    return found_cycle
                
                dfs(start_vertex)
                visited_global.add(start_vertex)
            
            for vertex in self._adjacency_list:
                if vertex not in visited_global:
                    find_cycles_from_vertex(vertex)
        
        else:
            # For undirected graphs: use DFS to find all elementary cycles
            visited_global = set()
            
            def find_cycles_from_vertex(start_vertex: Any):
                visited_local = set()
                
                def dfs(current: Any, path: List[Any], parent: Optional[Any] = None):
                    if current in path:
                        # Found a cycle
                        cycle_start_idx = path.index(current)
                        cycle = path[cycle_start_idx:] + [current]
                        if len(cycle) >= 4:  # At least 3 unique vertices + closing
                            cycles.append(cycle)
                        return
                    
                    if current in visited_local:
                        return
                    
                    visited_local.add(current)
                    path.append(current)
                    
                    neighbors = list(self._adjacency_list[current].keys()) if self._weighted else list(self._adjacency_list[current])
                    for neighbor in neighbors:
                        if neighbor != parent:  # Don't go back to parent immediately
                            dfs(neighbor, path, current)
                    
                    path.pop()
                
                dfs(start_vertex, [])
                visited_global.add(start_vertex)
            
            for vertex in self._adjacency_list:
                if vertex not in visited_global:
                    find_cycles_from_vertex(vertex)
        
        # Remove duplicate cycles
        unique_cycles = []
        seen_cycles = set()
        
        for cycle in cycles:
            if len(cycle) <= 1:
                continue
            
            # Normalize cycle: start with smallest vertex and choose direction with smaller lexicographic order
            cycle_vertices = cycle[:-1] if cycle[0] == cycle[-1] else cycle
            if not cycle_vertices:
                continue
                
            min_vertex = min(cycle_vertices)
            min_idx = cycle_vertices.index(min_vertex)
            
            # Create two possible normalizations (clockwise and counterclockwise)
            normalized1 = cycle_vertices[min_idx:] + cycle_vertices[:min_idx]
            normalized2 = (cycle_vertices[min_idx:] + cycle_vertices[:min_idx])[::-1]
            normalized2 = [normalized2[0]] + normalized2[1:][::-1]
            
            # Choose the lexicographically smaller one
            normalized = tuple(min(normalized1, normalized2))
            
            if normalized not in seen_cycles:
                seen_cycles.add(normalized)
                unique_cycles.append(list(normalized) + [normalized[0]])
        
        return unique_cycles
    
    def find_cycles(self) -> List[List[Any]]:
        """Find all simple cycles in the graph."""
        cycles = []
        visited = set()
        path_stack = []
        path_set = set()
        
        if self._directed:
            def dfs_find_cycles(vertex: Any):
                visited.add(vertex)
                path_stack.append(vertex)
                path_set.add(vertex)
                
                neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
                for neighbor in neighbors:
                    if neighbor in path_set:
                        cycle_start = path_stack.index(neighbor)
                        cycle = path_stack[cycle_start:]
                        if cycle not in cycles:
                            cycles.append(cycle[:])
                    elif neighbor not in visited:
                        dfs_find_cycles(neighbor)
                
                path_stack.pop()
                path_set.remove(vertex)
        else:
            def dfs_find_cycles(vertex: Any, parent: Optional[Any] = None):
                visited.add(vertex)
                path_stack.append(vertex)
                path_set.add(vertex)
                
                neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
                for neighbor in neighbors:
                    if neighbor == parent:
                        continue
                    if neighbor in path_set:
                        cycle_start = path_stack.index(neighbor)
                        cycle = path_stack[cycle_start:]
                        if len(cycle) >= 3 and cycle not in cycles:
                            cycles.append(cycle[:])
                    elif neighbor not in visited:
                        dfs_find_cycles(neighbor, vertex)
                
                path_stack.pop()
                path_set.remove(vertex)
        
        for vertex in self._adjacency_list:
            if vertex not in visited:
                dfs_find_cycles(vertex)
        
        return cycles
    
    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic."""
        return not self.has_cycle()
    
    # Connectivity methods
    def is_connected(self) -> bool:
        """
        Check if the graph is connected.
        For undirected: all vertices are reachable from any vertex.
        For directed: checks if strongly connected.
        """
        if not self._adjacency_list:
            return True
        
        if self._directed:
            return self.is_strongly_connected()
        
        # BFS from first vertex
        first_vertex = next(iter(self._adjacency_list))
        visited = set()
        stack = [first_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            
            neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return len(visited) == len(self._adjacency_list)
    
    def is_tree(self) -> bool:
        """
        Check if the graph is a tree.
        
        An undirected graph G with m edges and n nodes is a tree if ONE of:
        1. G is connected and m = n - 1
        2. G has no cycles and m = n - 1
        3. There is exactly one path between every pair of nodes
        
        A directed graph B with m edges and n nodes is a tree if:
        - The underlying undirected graph is a tree with root
        - There is exactly one path from the root to every other node
        
        Returns:
            bool: True if the graph is a tree, False otherwise
            
        Example:
            >>> # Undirected tree
            >>> g = Graph(directed=False)
            >>> g.add_edge(1, 2)
            >>> g.add_edge(1, 3)
            >>> g.add_edge(2, 4)
            >>> g.is_tree()
            True
            
            >>> # Add cycle - no longer a tree
            >>> g.add_edge(3, 4)
            >>> g.is_tree()
            False
        """
        if not self._adjacency_list:
            return True  # Empty graph is considered a tree
        
        n = len(self._adjacency_list)  # Number of nodes
        m = self.get_edge_count()      # Number of edges
        
        # Tree property: m = n - 1
        if m != n - 1:
            return False
        
        if self._directed:
            # For directed graph: check if it's a rooted tree
            # 1. Find potential root (node with in-degree 0)
            in_degree = {v: 0 for v in self._adjacency_list}
            
            for vertex in self._adjacency_list:
                neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
                for neighbor in neighbors:
                    in_degree[neighbor] += 1
            
            # Should have exactly one root (in-degree 0)
            roots = [v for v, deg in in_degree.items() if deg == 0]
            if len(roots) != 1:
                return False
            
            root = roots[0]
            
            # 2. Check if there's exactly one path from root to every other node
            # This is satisfied if:
            # - The graph is connected when treating edges as undirected (already checked via m = n-1)
            # - Every non-root node has exactly in-degree 1 (already verified)
            # - No cycles exist in directed sense
            
            # Check all non-root nodes have in-degree 1
            for vertex in self._adjacency_list:
                if vertex != root and in_degree[vertex] != 1:
                    return False
            
            # Check for cycles in directed graph
            if self.has_cycle():
                return False
            
            # Check connectivity from root (BFS)
            visited = set()
            queue = [root]
            
            while queue:
                vertex = queue.pop(0)
                if vertex in visited:
                    continue
                visited.add(vertex)
                
                neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            return len(visited) == n
        
        else:
            # For undirected graph: check if connected and acyclic
            # With m = n - 1 already verified, we just need to check connectivity
            # (because connected + m = n-1 guarantees no cycles)
            return self.is_connected()
    
    def get_edge_count(self) -> int:
        """
        Get the number of edges in the graph.
        For undirected graphs, counts each edge once.
        For directed graphs, counts each directed edge.
        
        Returns:
            int: Number of edges
        """
        if not self._adjacency_list:
            return 0
        
        count = 0
        if self._weighted:
            for vertex in self._adjacency_list:
                count += len(self._adjacency_list[vertex])
        else:
            for vertex in self._adjacency_list:
                count += len(self._adjacency_list[vertex])
        
        # For undirected graphs, each edge is counted twice
        if not self._directed:
            count //= 2
        
        return count
    
    def get_connected_components(self) -> List[Set[Any]]:
        """
        Get all connected components.
        For undirected graphs: returns connected components.
        For directed graphs: returns strongly connected components.
        """
        if self._directed:
            return self.get_strongly_connected_components()
        
        visited = set()
        components = []
        
        def dfs_component(vertex: Any, component: Set[Any]):
            visited.add(vertex)
            component.add(vertex)
            
            neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs_component(neighbor, component)
        
        for vertex in self._adjacency_list:
            if vertex not in visited:
                component = set()
                dfs_component(vertex, component)
                components.append(component)
        
        return components
    
    def is_strongly_connected(self) -> bool:
        """
        Check if the graph is strongly connected (only for directed graphs).
        For undirected graphs, equivalent to is_connected.
        """
        if not self._directed:
            return self.is_connected()
        
        if not self._adjacency_list:
            return True
        
        first_vertex = next(iter(self._adjacency_list))
        
        # Forward DFS
        visited = set()
        stack = [first_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            
            neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        if len(visited) != len(self._adjacency_list):
            return False
        
        # Reverse DFS
        reverse_adj = self._create_reverse_graph()
        visited = set()
        stack = [first_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            
            neighbors = list(reverse_adj[vertex].keys()) if self._weighted else list(reverse_adj[vertex])
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return len(visited) == len(self._adjacency_list)
    
    def get_strongly_connected_components(self) -> List[Set[Any]]:
        """
        Get all strongly connected components using Kosaraju's algorithm.
        Only for directed graphs. For undirected graphs, same as get_connected_components.
        """
        if not self._directed:
            return self.get_connected_components()
        
        if not self._adjacency_list:
            return []
        
        # Step 1: Fill order using DFS
        visited = set()
        finish_order = []
        
        def dfs_fill_order(vertex: Any):
            visited.add(vertex)
            neighbors = list(self._adjacency_list[vertex].keys()) if self._weighted else list(self._adjacency_list[vertex])
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs_fill_order(neighbor)
            finish_order.append(vertex)
        
        for vertex in self._adjacency_list:
            if vertex not in visited:
                dfs_fill_order(vertex)
        
        # Step 2: Create reverse graph
        reverse_adj = self._create_reverse_graph()
        
        # Step 3: DFS on reverse graph in reverse finish order
        visited = set()
        components = []
        
        def dfs_component(vertex: Any, component: Set[Any]):
            visited.add(vertex)
            component.add(vertex)
            neighbors = list(reverse_adj[vertex].keys()) if self._weighted else list(reverse_adj[vertex])
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs_component(neighbor, component)
        
        for vertex in reversed(finish_order):
            if vertex not in visited:
                component = set()
                dfs_component(vertex, component)
                components.append(component)
        
        return components
    
    def _create_reverse_graph(self) -> Dict:
        """Create a reversed version of the adjacency list (for directed graphs)."""
        if self._weighted:
            reverse_adj = {v: {} for v in self._adjacency_list}
            for vertex in self._adjacency_list:
                for neighbor, weight in self._adjacency_list[vertex].items():
                    reverse_adj[neighbor][vertex] = weight
        else:
            reverse_adj = {v: set() for v in self._adjacency_list}
            for vertex in self._adjacency_list:
                for neighbor in self._adjacency_list[vertex]:
                    reverse_adj[neighbor].add(vertex)
        return reverse_adj
    
    # Adjacency matrix methods
    def get_adjacency_matrix(self) -> List[List[Union[int, float]]]:
        """
        Get the adjacency matrix representation of the graph.
        For weighted graphs: matrix contains weights.
        For unweighted graphs: matrix contains 1 (edge exists) or 0 (no edge).
        """
        if not self._adjacency_list:
            return []
        
        vertices = sorted(self._adjacency_list.keys())
        n = len(vertices)
        vertex_to_index = {v: i for i, v in enumerate(vertices)}
        
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        for vertex in vertices:
            i = vertex_to_index[vertex]
            if self._weighted:
                for neighbor, weight in self._adjacency_list[vertex].items():
                    j = vertex_to_index[neighbor]
                    matrix[i][j] = weight
            else:
                for neighbor in self._adjacency_list[vertex]:
                    j = vertex_to_index[neighbor]
                    matrix[i][j] = 1
        
        return matrix
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: List[List[Union[int, float]]], 
                             vertices: Optional[List[Any]] = None,
                             directed: bool = False,
                             weighted: bool = False) -> 'Graph':
        """
        Create a graph from an adjacency matrix.
        
        Args:
            matrix: n×n adjacency matrix
            vertices: Optional list of vertex labels (uses 0 to n-1 if None)
            directed: Whether to create a directed graph
            weighted: Whether to interpret non-zero values as weights
            
        Returns:
            New Graph instance
        """
        if not matrix:
            return cls(directed=directed, weighted=weighted)
        
        n = len(matrix)
        if vertices is None:
            vertices = list(range(n))
        elif len(vertices) != n:
            raise ValueError("Number of vertices must match matrix dimensions")
        
        graph = cls(directed=directed, weighted=weighted)
        
        for vertex in vertices:
            graph.add_vertex(vertex)
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    if weighted:
                        graph.add_edge(vertices[i], vertices[j], matrix[i][j])
                    else:
                        graph.add_edge(vertices[i], vertices[j])
        
        return graph
    
    def get_adjacency_list(self) -> Dict[Any, Union[List[Any], List[tuple]]]:
        """
        Get a copy of the adjacency list representation of the graph.
        This dictionary can be used to create a new graph with the same structure.
        
        For unweighted graphs: Returns Dict[vertex, List[neighbor]]
        For weighted graphs: Returns Dict[vertex, List[(neighbor, weight)]]
        
        Returns:
            Dictionary representing the adjacency list of the graph
            
        Example:
            >>> g = Graph(directed=False, weighted=True)
            >>> g.add_edge('A', 'B', 10)
            >>> g.add_edge('B', 'C', 5)
            >>> adj_list = g.get_adjacency_list()
            >>> print(adj_list)
            {'A': [('B', 10)], 'B': [('A', 10), ('C', 5)], 'C': [('B', 5)]}
            >>> # Create new graph from adjacency list
            >>> g2 = Graph(directed=False, weighted=True, data=adj_list)
        """
        result = {}
        
        for vertex in self._adjacency_list:
            if self._weighted:
                # For weighted graphs: list of tuples (neighbor, weight)
                result[vertex] = [(neighbor, weight) for neighbor, weight in self._adjacency_list[vertex].items()]
            else:
                # For unweighted graphs: list of neighbors
                result[vertex] = list(self._adjacency_list[vertex])
        
        return result
    
    def visualize(self, title: Optional[str] = None, figsize: tuple = (12, 9), 
                 positions: Optional[Dict[Any, tuple]] = None):
        """
        Visualize the graph using matplotlib and networkx.
        
        Args:
            title: Title for the graph visualization
            figsize: Figure size as (width, height)
            positions: Optional dictionary mapping vertices to (x, y) coordinates
        """
        if not self._adjacency_list:
            print("Graph is empty - nothing to visualize")
            return
        
        # Create NetworkX graph
        if self._directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add vertices
        for vertex in self._adjacency_list:
            G.add_node(vertex)
        
        # Add edges
        if self._weighted:
            for v1, v2, weight in self.get_edges():
                G.add_edge(v1, v2, weight=weight)
        else:
            for edge in self.get_edges():
                G.add_edge(edge[0], edge[1])
        
        # Create visualization
        plt.figure(figsize=figsize)
        
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
                arrows=self._directed,
                arrowsize=20 if self._directed else 10,
                arrowstyle='->' if self._directed else '-')
        
        # Draw edge labels for weighted graphs
        if self._weighted:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, font_color='red')
        
        if title is None:
            graph_type = []
            if self._weighted:
                graph_type.append("Weighted")
            if self._directed:
                graph_type.append("Directed")
            else:
                graph_type.append("Undirected")
            title = " ".join(graph_type) + " Graph"
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """String representation of the graph."""
        if not self._adjacency_list:
            return "Empty graph"
        
        graph_type = []
        if self._weighted:
            graph_type.append("Weighted")
        if self._directed:
            graph_type.append("Directed")
        else:
            graph_type.append("Undirected")
        
        result = " ".join(graph_type) + " Graph:\n"
        for vertex in sorted(self._adjacency_list.keys()):
            if self._weighted:
                neighbors = sorted([(n, w) for n, w in self._adjacency_list[vertex].items()])
            else:
                neighbors = sorted(self._adjacency_list[vertex])
            result += f"  {vertex} -> {neighbors}\n"
        return result.rstrip()
    
    def __repr__(self) -> str:
        """Representation of the graph."""
        graph_type = "Weighted" if self._weighted else "Unweighted"
        graph_dir = "Directed" if self._directed else "Undirected"
        return f"Graph({graph_type}, {graph_dir}, vertices={self.vertex_count()}, edges={self.edge_count()})"


def main():
    """Test the unified Graph implementation in all 4 modes."""
    print("=== Testing Unified Graph Implementation ===\n")
    
    # Test 1: Undirected Unweighted Graph
    print("1. Undirected Unweighted Graph:")
    g1 = Graph(directed=False, weighted=False, data={'A': ['B', 'C'], 'B': ['C']})
    print(f"   {g1}")
    print(f"   Type: directed={g1.is_directed}, weighted={g1.is_weighted}")
    print(f"   Vertices: {g1.vertex_count()}, Edges: {g1.edge_count()}")
    print(f"   Has cycle: {g1.has_cycle()}\n")
    
    # Test 2: Directed Unweighted Graph
    print("2. Directed Unweighted Graph:")
    g2 = Graph(directed=True, weighted=False, data={'A': ['B'], 'B': ['C'], 'C': ['A']})
    print(f"   {g2}")
    print(f"   Type: directed={g2.is_directed}, weighted={g2.is_weighted}")
    print(f"   Vertices: {g2.vertex_count()}, Edges: {g2.edge_count()}")
    print(f"   Has cycle: {g2.has_cycle()}")
    print(f"   Is strongly connected: {g2.is_strongly_connected()}\n")
    
    # Test 3: Undirected Weighted Graph
    print("3. Undirected Weighted Graph:")
    g3 = Graph(directed=False, weighted=True, data={'A': [('B', 5), ('C', 3)], 'B': [('C', 2)]})
    print(f"   {g3}")
    print(f"   Type: directed={g3.is_directed}, weighted={g3.is_weighted}")
    print(f"   Vertices: {g3.vertex_count()}, Edges: {g3.edge_count()}")
    print(f"   Total weight: {g3.total_weight()}")
    path = g3.find_path('A', 'C')
    print(f"   Path A->C: {path}, Weight: {g3.path_weight(path)}\n")
    
    # Test 4: Directed Weighted Graph
    print("4. Directed Weighted Graph:")
    g4 = Graph(directed=True, weighted=True, data={'A': [('B', 10)], 'B': [('C', 20)], 'C': [('A', 5)]})
    print(f"   {g4}")
    print(f"   Type: directed={g4.is_directed}, weighted={g4.is_weighted}")
    print(f"   Vertices: {g4.vertex_count()}, Edges: {g4.edge_count()}")
    print(f"   Total weight: {g4.total_weight()}")
    print(f"   Weighted out-degree of A: {g4.weighted_out_degree('A')}")
    print(f"   Weighted in-degree of C: {g4.weighted_in_degree('C')}\n")
    
    # Test 5: Adjacency Matrix Conversion
    print("5. Adjacency Matrix Conversion:")
    matrix = g3.get_adjacency_matrix()
    print(f"   Matrix for undirected weighted graph:")
    for row in matrix:
        print(f"     {row}")
    
    g5 = Graph.from_adjacency_matrix(matrix, ['A', 'B', 'C'], directed=False, weighted=True)
    print(f"   Reconstructed graph: {g5}")
    print(f"   Edges match: {set(g3.get_edges()) == set(g5.get_edges())}\n")
    
    # Test 6: Graph operations
    print("6. Testing various graph operations:")
    g6 = Graph(directed=False, weighted=False)
    g6.add_vertex('X')
    g6.add_vertex('Y')
    g6.add_edge('X', 'Y')
    g6.add_edge('Y', 'Z')
    print(f"   Graph after adding vertices and edges: {g6}")
    print(f"   Neighbors of Y: {g6.get_neighbors('Y')}")
    print(f"   Degree of Y: {g6.degree('Y')}")
    print(f"   Is connected: {g6.is_connected()}")
    print(f"   Connected components: {g6.get_connected_components()}\n")
    
    # Test 7: Visualization
    print("7. Testing graph visualization:")
    # Create a simple graph for visualization
    g_vis = Graph(directed=False, weighted=True)
    g_vis.add_edge('A', 'B', 5)
    g_vis.add_edge('B', 'C', 3)
    g_vis.add_edge('C', 'D', 7)
    g_vis.add_edge('D', 'A', 2)
    g_vis.add_edge('B', 'D', 4)

    print(f"   Visualizing graph: {g_vis}")
    g_vis.visualize(title="Test Weighted Undirected Graph")

    # Test directed graph visualization
    g_vis_dir = Graph(directed=True, weighted=False)
    g_vis_dir.add_edge('Start', 'Middle')
    g_vis_dir.add_edge('Middle', 'End')
    g_vis_dir.add_edge('Start', 'End')
    print(f"   Visualizing directed graph: {g_vis_dir}")
    print("   Displaying directed graph visualization...")
    g_vis_dir.visualize(title="Test Directed Graph", figsize=(10, 8))
    
    print("=== All tests completed ===")


if __name__ == "__main__":
    main()
