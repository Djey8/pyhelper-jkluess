import pytest
from Complex.Graphs.weighted_directed_graph import WeightedDirectedGraph


class TestWeightedDirectedGraphCreation:
    def test_empty_graph_creation(self):
        """Test creating an empty weighted directed graph"""
        graph = WeightedDirectedGraph()
        assert graph.vertex_count() == 0
        assert graph.edge_count() == 0
        assert graph.get_vertices() == []
        assert graph.get_edges() == []
    
    def test_graph_creation_with_tuple_data(self):
        """Test creating a graph with initial weighted data using tuples"""
        data = {
            'A': [('B', 5), ('C', 10)],
            'B': [('C', 3)],
            'C': []
        }
        graph = WeightedDirectedGraph(data)
        assert graph.vertex_count() == 3
        assert graph.edge_count() == 3


class TestWeightedDirectedVertexOperations:
    def test_add_single_vertex(self):
        """Test adding a single vertex"""
        graph = WeightedDirectedGraph()
        result = graph.add_vertex('A')
        assert result is True
        assert graph.has_vertex('A')
        assert graph.vertex_count() == 1
    
    def test_add_duplicate_vertex(self):
        """Test adding a duplicate vertex returns False"""
        graph = WeightedDirectedGraph()
        graph.add_vertex('A')
        result = graph.add_vertex('A')
        assert result is False
        assert graph.vertex_count() == 1
    
    def test_remove_vertex_with_edges(self):
        """Test removing a vertex removes all associated edges"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 3)
        graph.add_edge('C', 'A', 7)
        
        result = graph.remove_vertex('B')
        assert result is True
        assert not graph.has_vertex('B')
        assert graph.vertex_count() == 2
        assert graph.edge_count() == 1  # Only C->A remains


class TestWeightedDirectedEdgeOperations:
    def test_add_weighted_directed_edge(self):
        """Test adding a weighted directed edge"""
        graph = WeightedDirectedGraph()
        result = graph.add_edge('A', 'B', 10)
        
        assert result is True
        assert graph.has_edge('A', 'B')
        assert not graph.has_edge('B', 'A')  # Directed!
        assert graph.get_edge_weight('A', 'B') == 10
    
    def test_add_edge_default_weight(self):
        """Test adding edge with default weight"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B')  # No weight specified
        
        assert graph.has_edge('A', 'B')
        assert graph.get_edge_weight('A', 'B') == 1  # Default weight
    
    def test_update_edge_weight(self):
        """Test updating edge weight"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        result = graph.update_edge_weight('A', 'B', 10)  # Update weight
        
        assert result is True
        assert graph.get_edge_weight('A', 'B') == 10
        assert graph.edge_count() == 1  # Still one edge
    
    def test_bidirectional_edges_different_weights(self):
        """Test that A->B and B->A can have different weights"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'A', 10)
        
        assert graph.get_edge_weight('A', 'B') == 5
        assert graph.get_edge_weight('B', 'A') == 10
        assert graph.edge_count() == 2
    
    def test_remove_directed_edge(self):
        """Test removing a directed edge"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'A', 10)
        
        result = graph.remove_edge('A', 'B')
        assert result is True
        assert not graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'A')  # Other direction still exists
        assert graph.edge_count() == 1
    
    def test_get_weight_nonexistent_edge(self):
        """Test getting weight of non-existent edge"""
        graph = WeightedDirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        
        assert graph.get_edge_weight('A', 'B') is None


class TestWeightedDirectedGraphDegrees:
    def test_out_degree(self):
        """Test out-degree calculation"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('A', 'C', 3)
        graph.add_edge('A', 'D', 7)
        
        assert graph.out_degree('A') == 3
        assert graph.out_degree('B') == 0
    
    def test_in_degree(self):
        """Test in-degree calculation"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'D', 5)
        graph.add_edge('B', 'D', 3)
        graph.add_edge('C', 'D', 7)
        
        assert graph.in_degree('D') == 3
        assert graph.in_degree('A') == 0
    
    def test_total_degree(self):
        """Test total degree (in + out)"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 3)
        graph.add_edge('C', 'A', 7)
        
        # Each vertex has in-degree 1 and out-degree 1
        assert graph.degree('A') == 2
        assert graph.degree('B') == 2
        assert graph.degree('C') == 2


class TestWeightedDirectedGraphQueries:
    def test_get_weighted_neighbors(self):
        """Test getting successors with their weights"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('A', 'C', 10)
        
        neighbors = graph.get_weighted_neighbors('A')
        assert set(neighbors) == {('B', 5), ('C', 10)}
    
    def test_get_neighbors_without_weights(self):
        """Test getting just the successor vertices"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('A', 'C', 10)
        
        neighbors = graph.get_neighbors('A')
        assert set(neighbors) == {'B', 'C'}
    
    def test_get_predecessors(self):
        """Test getting predecessor vertices"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'C', 5)
        graph.add_edge('B', 'C', 10)
        
        predecessors = graph.get_predecessors('C')
        assert set(predecessors) == {'A', 'B'}
    
    def test_get_weighted_predecessors(self):
        """Test getting predecessors with their weights"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'C', 5)
        graph.add_edge('B', 'C', 10)
        
        predecessors = graph.get_weighted_predecessors('C')
        assert set(predecessors) == {('A', 5), ('B', 10)}


# TODO: Implement Dijkstra's algorithm in WeightedDirectedGraph
# class TestDijkstraDirected:
#     def test_dijkstra_simple_directed_path(self):
#         """Test Dijkstra's algorithm on a simple directed path"""
#         graph = WeightedDirectedGraph()
#         graph.add_edge('A', 'B', 5)
#         graph.add_edge('B', 'C', 3)
#         
#         path, distance = graph.dijkstra('A', 'C')
#         assert path == ['A', 'B', 'C']
#         assert distance == 8


class TestWeightedDirectedGraphTheory:
    def test_is_simple_graph(self):
        """Test simple graph detection (no self-loops)"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 3)
        
        assert graph.is_simple_graph() is True
        
        # Add self-loop
        graph.add_edge('A', 'A', 2)
        assert graph.is_simple_graph() is False
    
    def test_get_degree_sequence(self):
        """Test getting degree sequences"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 3)
        graph.add_edge('C', 'A', 7)
        
        sequences = graph.get_degree_sequence()
        assert 'in_degrees' in sequences
        assert 'out_degrees' in sequences
        assert 'total_degrees' in sequences
        # All vertices have in=1, out=1, total=2
        assert sequences['in_degrees'] == [1, 1, 1]
        assert sequences['out_degrees'] == [1, 1, 1]
        assert sequences['total_degrees'] == [2, 2, 2]
    
    def test_get_graph_info(self):
        """Test comprehensive graph information"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('A', 'C', 3)
        graph.add_edge('B', 'C', 7)
        
        info = graph.get_graph_info()
        assert info['vertices'] == 3
        assert info['edges'] == 3
        assert info['is_simple'] is True
        # Check degree sequences exist
        assert 'degree_sequences' in info
        assert info['degree_sequences']['out_degrees'][0] == 2  # A has 2 outgoing


class TestWeightedDirectedEdgeCases:
    def test_negative_weights_dijkstra(self):
        """Test graph with negative weights"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', -2)
        
        assert graph.get_edge_weight('B', 'C') == -2
        # TODO: Test pathfinding when dijkstra is implemented
    
    def test_self_loop_weighted(self):
        """Test self-loop with weight"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'A', 5)
        
        assert graph.has_edge('A', 'A')
        assert graph.get_edge_weight('A', 'A') == 5
        assert graph.in_degree('A') == 1
        assert graph.out_degree('A') == 1
    
    def test_zero_weight_edge(self):
        """Test edge with zero weight"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 0)
        
        assert graph.get_edge_weight('A', 'B') == 0
        assert graph.has_edge('A', 'B')
    
    def test_float_weights(self):
        """Test edges with float weights"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 3.5)
        graph.add_edge('B', 'C', 2.7)
        
        assert graph.get_edge_weight('A', 'B') == 3.5
        assert graph.get_edge_weight('B', 'C') == 2.7


class TestWeightedDirectedGraphString:
    def test_string_representation(self):
        """Test string representation"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        
        string_repr = str(graph)
        assert 'Weighted Directed Graph' in string_repr
        assert 'A' in string_repr
        assert 'B' in string_repr
    
    def test_repr(self):
        """Test repr"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 3)
        
        repr_str = repr(graph)
        assert 'WeightedDirectedGraph' in repr_str
        assert 'vertices=3' in repr_str
        assert 'edges=2' in repr_str


class TestWeightedDirectedRealWorld:
    def test_task_scheduling(self):
        """Test task scheduling with durations"""
        graph = WeightedDirectedGraph()
        # Tasks with durations (in hours)
        graph.add_edge('Start', 'TaskA', 2)
        graph.add_edge('Start', 'TaskB', 3)
        graph.add_edge('TaskA', 'TaskC', 4)
        graph.add_edge('TaskB', 'TaskC', 2)
        graph.add_edge('TaskC', 'End', 1)
        
        # Verify graph structure
        assert graph.out_degree('Start') == 2
        assert graph.in_degree('TaskC') == 2
        assert graph.vertex_count() == 5
    
    def test_flight_routes(self):
        """Test one-way flight routes with costs"""
        graph = WeightedDirectedGraph()
        graph.add_edge('NYC', 'LON', 500)
        graph.add_edge('NYC', 'PAR', 600)
        graph.add_edge('LON', 'BER', 150)
        graph.add_edge('PAR', 'BER', 100)
        
        # Verify route structure
        assert graph.get_edge_weight('NYC', 'LON') == 500
        assert graph.get_edge_weight('LON', 'BER') == 150
        assert graph.out_degree('NYC') == 2
    
    def test_web_page_links(self):
        """Test web page link structure"""
        graph = WeightedDirectedGraph()
        # Pages with link weights (importance/click-through rate)
        graph.add_edge('Home', 'Products', 10)
        graph.add_edge('Home', 'About', 5)
        graph.add_edge('Products', 'Details', 8)
        graph.add_edge('About', 'Contact', 3)
        
        assert graph.out_degree('Home') == 2
        assert graph.in_degree('Details') == 1
        assert graph.get_edge_weight('Home', 'Products') == 10


class TestWeightedDirectedGraphPaths:
    def test_find_path(self):
        """Test finding a directed path (ignoring weights)"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        
        path = graph.find_path('A', 'C')
        assert path is not None
        assert path == ['A', 'B', 'C']
        
        # Reverse direction should not work
        path = graph.find_path('C', 'A')
        assert path is None
    
    def test_path_weight(self):
        """Test calculating path weight"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        graph.add_edge('C', 'D', 3)
        
        path = ['A', 'B', 'C', 'D']
        weight = graph.path_weight(path)
        assert weight == 18  # 5 + 10 + 3
    
    def test_is_reachable_directed(self):
        """Test reachability respects direction"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        
        assert graph.is_reachable('A', 'C') is True
        assert graph.is_reachable('C', 'A') is False


class TestWeightedDirectedGraphCycles:
    def test_has_cycle(self):
        """Test cycle detection in weighted directed graph"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        graph.add_edge('C', 'A', 3)
        
        assert graph.has_cycle() is True
    
    def test_is_acyclic_dag(self):
        """Test weighted DAG"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('A', 'C', 10)
        graph.add_edge('B', 'D', 3)
        graph.add_edge('C', 'D', 7)
        
        assert graph.is_acyclic() is True


class TestWeightedDirectedGraphConnectivity:
    def test_is_strongly_connected(self):
        """Test strong connectivity in weighted graph"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        graph.add_edge('C', 'A', 3)
        
        assert graph.is_strongly_connected() is True
    
    def test_get_strongly_connected_components(self):
        """Test strongly connected components"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'A', 10)
        graph.add_edge('C', 'D', 3)
        graph.add_edge('D', 'C', 7)
        
        components = graph.get_strongly_connected_components()
        assert len(components) == 2


class TestWeightedDirectedGraphAdjacencyMatrix:
    def test_get_adjacency_matrix(self):
        """Test adjacency matrix with weights for directed graph"""
        graph = WeightedDirectedGraph()
        graph.add_edge('A', 'B', 5)
        graph.add_edge('B', 'C', 10)
        
        matrix = graph.get_adjacency_matrix()
        assert len(matrix) == 3
        # Check weights are preserved
        # Matrix should not be symmetric for directed graphs
    
    def test_from_adjacency_matrix(self):
        """Test creating weighted directed graph from matrix"""
        matrix = [
            [0, 5, 0],
            [0, 0, 10],
            [3, 0, 0]
        ]
        vertices = ['A', 'B', 'C']
        
        graph = WeightedDirectedGraph.from_adjacency_matrix(matrix, vertices)
        assert graph.vertex_count() == 3
        assert graph.get_edge_weight('A', 'B') == 5
        assert graph.get_edge_weight('B', 'C') == 10
        assert graph.get_edge_weight('C', 'A') == 3
        # Check direction matters
        assert graph.get_edge_weight('B', 'A') is None
    
    def test_adjacency_matrix_roundtrip(self):
        """Test converting to and from weighted adjacency matrix"""
        original = WeightedDirectedGraph()
        original.add_edge('A', 'B', 5)
        original.add_edge('B', 'C', 10)
        
        matrix = original.get_adjacency_matrix()
        reconstructed = WeightedDirectedGraph.from_adjacency_matrix(matrix, ['A', 'B', 'C'])
        
        assert original.vertex_count() == reconstructed.vertex_count()
        assert original.edge_count() == reconstructed.edge_count()
        assert reconstructed.get_edge_weight('A', 'B') == 5
        assert reconstructed.get_edge_weight('B', 'C') == 10
