import pytest
from pyhelper_jkluess.Complex.Graphs.weighted_undirected_graph import WeightedUndirectedGraph


# Note: Dijkstra pathfinding is not yet implemented in WeightedUndirectedGraph
# Tests for dijkstra() method are commented out and can be enabled when implemented


class TestWeightedUndirectedGraphCreation:
    def test_empty_graph_creation(self):
        """Test creating an empty weighted undirected graph"""
        graph = WeightedUndirectedGraph()
        assert graph.vertex_count() == 0
        assert graph.edge_count() == 0
        assert graph.get_vertices() == []
        assert graph.get_edges() == []
    
    def test_graph_creation_with_data(self):
        """Test creating a graph with initial weighted data"""
        data = {
            'A': {'B': 5, 'C': 10},
            'B': {'A': 5, 'C': 3},
            'C': {'A': 10, 'B': 3}
        }
        graph = WeightedUndirectedGraph(data)
        assert graph.vertex_count() == 3
        assert graph.edge_count() == 3


class TestWeightedVertexOperations:
    def test_add_single_vertex(self):
        """Test adding a single vertex"""
        graph = WeightedUndirectedGraph()
        result = graph.add_vertex('A')
        assert result is True
        assert graph.has_vertex('A')
        assert graph.vertex_count() == 1
    
    def test_add_duplicate_vertex(self):
        """Test adding a duplicate vertex returns False"""
        graph = WeightedUndirectedGraph()
        graph.add_vertex('A')
        result = graph.add_vertex('A')
        assert result is False
        assert graph.vertex_count() == 1
    
    def test_remove_vertex(self):
        """Test removing a vertex"""
        graph = WeightedUndirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        graph.add_edge('A', 'B', 5)
        
        result = graph.remove_vertex('A')
        assert result is True
        assert not graph.has_vertex('A')
        assert graph.vertex_count() == 1
        assert graph.edge_count() == 0


class TestWeightedEdgeOperations:
    def test_add_weighted_edge(self):
        """Test adding a weighted edge"""
        graph = WeightedUndirectedGraph()
        result = graph.add_edge('A', 'B', 10)
        
        assert result is True
        assert graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'A')  # Undirected
        assert graph.get_edge_weight('A', 'B') == 10
        assert graph.get_edge_weight('B', 'A') == 10
    
    def test_add_edge_default_weight(self):
        """Test adding edge with default weight"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B')  # No weight specified
        
        assert graph.has_edge('A', 'B')
        assert graph.get_edge_weight('A', 'B') == 1  # Default weight
    
    def test_update_edge_weight(self):
        """Test updating an existing edge weight"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        result = graph.update_edge_weight('A', 'B', 10)  # Update weight
        
        assert result is True
        assert graph.get_edge_weight('A', 'B') == 10
        assert graph.edge_count() == 1  # Still one edge
    
    def test_remove_weighted_edge(self):
        """Test removing a weighted edge"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        
        result = graph.remove_edge('A', 'B')
        assert result is True
        assert not graph.has_edge('A', 'B')
        assert not graph.has_edge('B', 'A')
        assert graph.edge_count() == 0
    
    def test_get_weight_nonexistent_edge(self):
        """Test getting weight of non-existent edge"""
        graph = WeightedUndirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        
        assert graph.get_edge_weight('A', 'B') is None
    
    def test_multiple_weighted_edges(self):
        """Test adding multiple weighted edges"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 3)
        graph.add_edge('C', 'A', 7)
        
        assert graph.edge_count() == 3
        assert graph.get_edge_weight('A', 'B') == 5
        assert graph.get_edge_weight('B', 'C') == 3
        assert graph.get_edge_weight('C', 'A') == 7


class TestWeightedGraphQueries:
    def test_get_weighted_neighbors(self):
        """Test getting neighbors with their weights"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('A', 'C', 10)
        
        neighbors = graph.get_weighted_neighbors('A')
        assert neighbors == {'B': 5, 'C': 10}
    
    def test_get_neighbors_without_weights(self):
        """Test getting just the neighbor vertices"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('A', 'C', 10)
        
        neighbors = graph.get_neighbors('A')
        assert set(neighbors) == {'B', 'C'}
    
    def test_get_edges_with_weights(self):
        """Test getting all edges with weights"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 3)
        
        edges = graph.get_edges()
        assert len(edges) == 2
        # Check edges (order may vary)
        edge_set = {(min(e[0], e[1]), max(e[0], e[1]), e[2]) for e in edges}
        assert ('A', 'B', 5) in edge_set
        assert ('B', 'C', 3) in edge_set


# TODO: Implement Dijkstra's algorithm in WeightedUndirectedGraph
# class TestDijkstraAlgorithm:
#     def test_dijkstra_simple_path(self):
#         """Test Dijkstra's algorithm on a simple path"""
#         graph = WeightedUndirectedGraph()
#         graph.add_edge('A', 'B', 5)
#         graph.add_edge('B', 'C', 3)
#         
#         path, distance = graph.dijkstra('A', 'C')
#         assert path == ['A', 'B', 'C']
#         assert distance == 8


class TestWeightedGraphDegree:
    def test_degree_simple(self):
        """Test degree calculation in weighted graph"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('A', 'C', 3)
        graph.add_edge('A', 'D', 7)
        
        assert graph.degree('A') == 3
        assert graph.degree('B') == 1
    
    def test_degree_isolated_vertex(self):
        """Test degree of isolated vertex"""
        graph = WeightedUndirectedGraph()
        graph.add_vertex('A')
        
        assert graph.degree('A') == 0


class TestWeightedGraphEdgeCases:
    def test_negative_weights(self):
        """Test graph with negative weights"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', -5)
        
        assert graph.get_edge_weight('A', 'B') == -5
    
    def test_zero_weight(self):
        """Test edge with zero weight"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 0)
        
        assert graph.get_edge_weight('A', 'B') == 0
        assert graph.has_edge('A', 'B')
    
    def test_float_weights(self):
        """Test edges with float weights"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 3.5)
        graph.add_edge('B', 'C', 2.7)
        
        assert graph.get_edge_weight('A', 'B') == 3.5
        assert graph.get_edge_weight('B', 'C') == 2.7
        # TODO: Test pathfinding when dijkstra is implemented
        # path, distance = graph.dijkstra('A', 'C')
        # assert abs(distance - 6.2) < 0.001  # Float comparison
    
    def test_self_loop_weighted(self):
        """Test self-loop with weight"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'A', 5)
        
        assert graph.has_edge('A', 'A')
        assert graph.get_edge_weight('A', 'A') == 5
    
    def test_large_weights(self):
        """Test edges with large weights"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 1000000)
        
        assert graph.get_edge_weight('A', 'B') == 1000000


class TestWeightedGraphString:
    def test_string_representation(self):
        """Test string representation of weighted graph"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        
        string_repr = str(graph)
        assert 'Weighted Undirected Graph' in string_repr
        assert 'A' in string_repr
        assert 'B' in string_repr
    
    def test_repr(self):
        """Test repr of weighted graph"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 3)
        
        repr_str = repr(graph)
        assert 'WeightedUndirectedGraph' in repr_str
        assert 'vertices=3' in repr_str
        assert 'edges=2' in repr_str


class TestWeightedGraphRealWorld:
    def test_road_network(self):
        """Test modeling a simple road network"""
        graph = WeightedUndirectedGraph()
        # Cities with distances in km
        graph.add_edge('Berlin', 'Munich', 584)
        graph.add_edge('Munich', 'Vienna', 434)
        graph.add_edge('Berlin', 'Vienna', 680)
        graph.add_edge('Berlin', 'Hamburg', 289)
        
        # Verify graph structure
        assert graph.has_edge('Berlin', 'Munich')
        assert graph.get_edge_weight('Berlin', 'Vienna') == 680
        assert graph.vertex_count() == 4
        assert graph.edge_count() == 4
        # TODO: Test pathfinding when dijkstra is implemented
        # path, distance = graph.dijkstra('Berlin', 'Vienna')
    
    def test_social_network_weighted(self):
        """Test weighted social network (connection strength)"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('Alice', 'Bob', 8)      # Strong connection
        graph.add_edge('Bob', 'Charlie', 5)    # Medium connection
        graph.add_edge('Alice', 'Charlie', 3)  # Weak connection
        
        # Verify connections
        assert graph.get_edge_weight('Alice', 'Charlie') == 3
        assert graph.degree('Bob') == 2
        # TODO: Test pathfinding when dijkstra is implemented
        # path, strength = graph.dijkstra('Alice', 'Charlie')


class TestWeightedUndirectedGraphPaths:
    def test_find_path(self):
        """Test finding a path (ignoring weights)"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        
        path = graph.find_path('A', 'C')
        assert path is not None
        assert path[0] == 'A'
        assert path[-1] == 'C'
    
    def test_path_weight(self):
        """Test calculating path weight"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        graph.add_edge('C', 'D', 3)
        
        path = ['A', 'B', 'C', 'D']
        weight = graph.path_weight(path)
        assert weight == 18  # 5 + 10 + 3
    
    def test_path_weight_invalid(self):
        """Test path weight for invalid path"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        
        path = ['A', 'C']
        assert graph.path_weight(path) == 0


class TestWeightedUndirectedGraphCycles:
    def test_has_cycle(self):
        """Test cycle detection in weighted graph"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        graph.add_edge('C', 'A', 3)
        
        assert graph.has_cycle() is True
    
    def test_is_acyclic(self):
        """Test acyclic weighted graph"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        
        assert graph.is_acyclic() is True


class TestWeightedUndirectedGraphConnectivity:
    def test_is_connected(self):
        """Test connectivity in weighted graph"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        
        assert graph.is_connected() is True
    
    def test_get_connected_components(self):
        """Test connected components"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('C', 'D', 10)
        
        components = graph.get_connected_components()
        assert len(components) == 2


class TestWeightedUndirectedGraphAdjacencyMatrix:
    def test_get_adjacency_matrix(self):
        """Test adjacency matrix with weights"""
        graph = WeightedUndirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        
        matrix = graph.get_adjacency_matrix()
        assert len(matrix) == 3
        # Check weights are preserved
        assert any(5 in row for row in matrix)
        assert any(10 in row for row in matrix)
    
    def test_from_adjacency_matrix(self):
        """Test creating weighted graph from matrix"""
        matrix = [
            [0, 5, 10],
            [5, 0, 3],
            [10, 3, 0]
        ]
        vertices = ['A', 'B', 'C']
        
        graph = WeightedUndirectedGraph.from_adjacency_matrix(matrix, vertices)
        assert graph.vertex_count() == 3
        assert graph.get_edge_weight('A', 'B') == 5
        assert graph.get_edge_weight('B', 'C') == 3
        assert graph.get_edge_weight('A', 'C') == 10
    
    def test_adjacency_matrix_roundtrip(self):
        """Test converting to and from weighted adjacency matrix"""
        original = WeightedUndirectedGraph()
        original.add_edge('A', 'B', 5)
        original.add_edge('B', 'C', 10)
        
        matrix = original.get_adjacency_matrix()
        reconstructed = WeightedUndirectedGraph.from_adjacency_matrix(matrix, ['A', 'B', 'C'])
        
        assert original.vertex_count() == reconstructed.vertex_count()
        assert original.edge_count() == reconstructed.edge_count()
        assert reconstructed.get_edge_weight('A', 'B') == 5
        assert reconstructed.get_edge_weight('B', 'C') == 10
