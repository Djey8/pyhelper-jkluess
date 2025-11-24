import pytest
from pyhelper_jkluess.Complex.Graphs.undirected_graph import UndirectedGraph


class TestUndirectedGraphCreation:
    def test_empty_graph_creation(self):
        """Test creating an empty undirected graph"""
        graph = UndirectedGraph()
        assert graph.vertex_count() == 0
        assert graph.edge_count() == 0
        assert graph.get_vertices() == []
        assert graph.get_edges() == []
    
    def test_graph_creation_with_data(self):
        """Test creating a graph with initial data"""
        data = {
            'A': ['B', 'C'],
            'B': ['A', 'C'],
            'C': ['A', 'B']
        }
        graph = UndirectedGraph(data)
        assert graph.vertex_count() == 3
        assert graph.edge_count() == 3


class TestVertexOperations:
    def test_add_single_vertex(self):
        """Test adding a single vertex"""
        graph = UndirectedGraph()
        result = graph.add_vertex('A')
        assert result is True
        assert graph.has_vertex('A')
        assert graph.vertex_count() == 1
    
    def test_add_duplicate_vertex(self):
        """Test adding a duplicate vertex returns False"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        result = graph.add_vertex('A')
        assert result is False
        assert graph.vertex_count() == 1
    
    def test_add_multiple_vertices(self):
        """Test adding multiple vertices"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        graph.add_vertex('C')
        assert graph.vertex_count() == 3
        assert set(graph.get_vertices()) == {'A', 'B', 'C'}
    
    def test_remove_vertex(self):
        """Test removing a vertex"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        graph.add_edge('A', 'B')
        
        result = graph.remove_vertex('A')
        assert result is True
        assert not graph.has_vertex('A')
        assert graph.vertex_count() == 1
        assert graph.edge_count() == 0
    
    def test_remove_nonexistent_vertex(self):
        """Test removing a vertex that doesn't exist"""
        graph = UndirectedGraph()
        result = graph.remove_vertex('Z')
        assert result is False
    
    def test_remove_vertex_removes_all_edges(self):
        """Test that removing a vertex removes all its edges"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'C')
        
        graph.remove_vertex('A')
        assert not graph.has_edge('A', 'B')
        assert not graph.has_edge('A', 'C')
        assert graph.has_edge('B', 'C')
        assert graph.edge_count() == 1


class TestEdgeOperations:
    def test_add_edge_between_existing_vertices(self):
        """Test adding an edge between existing vertices"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        result = graph.add_edge('A', 'B')
        
        assert result is True
        assert graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'A')  # Undirected
        assert graph.edge_count() == 1
    
    def test_add_edge_creates_vertices(self):
        """Test that adding edge creates vertices if they don't exist"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        
        assert graph.has_vertex('A')
        assert graph.has_vertex('B')
        assert graph.has_edge('A', 'B')
    
    def test_add_duplicate_edge(self):
        """Test adding a duplicate edge returns False"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        result = graph.add_edge('A', 'B')
        
        assert result is False
        assert graph.edge_count() == 1
    
    def test_remove_edge(self):
        """Test removing an edge"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        result = graph.remove_edge('A', 'B')
        
        assert result is True
        assert not graph.has_edge('A', 'B')
        assert not graph.has_edge('B', 'A')
        assert graph.edge_count() == 0
    
    def test_remove_nonexistent_edge(self):
        """Test removing an edge that doesn't exist"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        result = graph.remove_edge('A', 'B')
        
        assert result is False
    
    def test_edge_symmetry(self):
        """Test that edges are symmetric in undirected graph"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        
        assert 'B' in graph.get_neighbors('A')
        assert 'A' in graph.get_neighbors('B')


class TestGraphQueries:
    def test_get_neighbors_empty(self):
        """Test getting neighbors of isolated vertex"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        assert graph.get_neighbors('A') == []
    
    def test_get_neighbors(self):
        """Test getting neighbors of a vertex"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('A', 'D')
        
        neighbors = graph.get_neighbors('A')
        assert set(neighbors) == {'B', 'C', 'D'}
    
    def test_get_neighbors_nonexistent_vertex(self):
        """Test getting neighbors of nonexistent vertex"""
        graph = UndirectedGraph()
        assert graph.get_neighbors('Z') == []
    
    def test_get_vertices(self):
        """Test getting all vertices"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        graph.add_vertex('C')
        
        vertices = graph.get_vertices()
        assert set(vertices) == {'A', 'B', 'C'}
    
    def test_get_edges(self):
        """Test getting all edges"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('A', 'C')
        
        edges = graph.get_edges()
        assert len(edges) == 3
        # Check edges are present (order doesn't matter)
        edge_set = {tuple(sorted(edge)) for edge in edges}
        expected = {('A', 'B'), ('A', 'C'), ('B', 'C')}
        assert edge_set == expected
    
    def test_has_vertex(self):
        """Test checking vertex existence"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        
        assert graph.has_vertex('A')
        assert not graph.has_vertex('B')
    
    def test_has_edge(self):
        """Test checking edge existence"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        
        assert graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'A')
        assert not graph.has_edge('A', 'C')


class TestGraphProperties:
    def test_vertex_count(self):
        """Test counting vertices"""
        graph = UndirectedGraph()
        assert graph.vertex_count() == 0
        
        graph.add_vertex('A')
        assert graph.vertex_count() == 1
        
        graph.add_vertex('B')
        graph.add_vertex('C')
        assert graph.vertex_count() == 3
    
    def test_edge_count(self):
        """Test counting edges"""
        graph = UndirectedGraph()
        assert graph.edge_count() == 0
        
        graph.add_edge('A', 'B')
        assert graph.edge_count() == 1
        
        graph.add_edge('B', 'C')
        graph.add_edge('A', 'C')
        assert graph.edge_count() == 3


class TestGraphWithDifferentDataTypes:
    def test_integer_vertices(self):
        """Test graph with integer vertices"""
        graph = UndirectedGraph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        
        assert graph.has_vertex(1)
        assert graph.has_edge(1, 2)
        assert set(graph.get_neighbors(2)) == {1, 3}
    
    def test_mixed_type_vertices(self):
        """Test graph with mixed type vertices"""
        graph = UndirectedGraph()
        graph.add_edge('A', 1)
        graph.add_edge(1, 2.5)
        
        assert graph.has_vertex('A')
        assert graph.has_vertex(1)
        assert graph.has_vertex(2.5)
        assert graph.has_edge('A', 1)


class TestGraphStringRepresentation:
    def test_str_empty_graph(self):
        """Test string representation of empty graph"""
        graph = UndirectedGraph()
        assert str(graph) == "Empty graph"
    
    def test_str_graph_with_data(self):
        """Test string representation of graph with data"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        result = str(graph)
        
        assert "Undirected Graph:" in result
        assert "A:" in result
        assert "B:" in result
    
    def test_repr_graph(self):
        """Test repr of graph"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        result = repr(graph)
        assert "UndirectedGraph" in result
        assert "vertices=3" in result
        assert "edges=2" in result


class TestComplexGraphScenarios:
    def test_disconnected_graph(self):
        """Test graph with disconnected components"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('C', 'D')
        
        assert graph.vertex_count() == 4
        assert graph.edge_count() == 2
        assert not graph.has_edge('A', 'C')
    
    def test_complete_graph(self):
        """Test complete graph (all vertices connected)"""
        graph = UndirectedGraph()
        vertices = ['A', 'B', 'C', 'D']
        
        # Add all possible edges
        for i, v1 in enumerate(vertices):
            for v2 in vertices[i+1:]:
                graph.add_edge(v1, v2)
        
        assert graph.vertex_count() == 4
        assert graph.edge_count() == 6  # n(n-1)/2 = 4*3/2 = 6
    
    def test_star_graph(self):
        """Test star graph (one central vertex connected to all others)"""
        graph = UndirectedGraph()
        center = 'Center'
        for i in range(5):
            graph.add_edge(center, f'Node{i}')
        
        assert graph.vertex_count() == 6
        assert graph.edge_count() == 5
        assert len(graph.get_neighbors(center)) == 5
    
    def test_self_loop_prevention(self):
        """Test that adding edge from vertex to itself works"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'A')
        
        assert graph.has_vertex('A')
        # Self-loop should be counted
        assert graph.has_edge('A', 'A')


class TestUndirectedGraphTheoryMethods:
    """Test new graph theory analysis methods"""
    
    def test_degree_calculation(self):
        """Test degree calculation"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('A', 'D')
        
        assert graph.degree('A') == 3
        assert graph.degree('B') == 1
        assert graph.degree('C') == 1
        assert graph.degree('D') == 1
    
    def test_degree_complete_graph(self):
        """Test degree in complete graph"""
        graph = UndirectedGraph()
        vertices = ['A', 'B', 'C', 'D']
        
        # Complete graph - every vertex connected to every other
        for i, v1 in enumerate(vertices):
            for v2 in vertices[i+1:]:
                graph.add_edge(v1, v2)
        
        # In complete graph K4, every vertex has degree 3
        for v in vertices:
            assert graph.degree(v) == 3
    
    def test_degree_isolated_vertex(self):
        """Test degree of isolated vertex"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        assert graph.degree('A') == 0
    
    def test_degree_nonexistent_vertex(self):
        """Test degree of non-existent vertex"""
        graph = UndirectedGraph()
        assert graph.degree('Z') == 0
    
    def test_is_simple_graph_true(self):
        """Test is_simple_graph returns True for graph without self-loops"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        assert graph.is_simple_graph() is True
    
    def test_is_simple_graph_false(self):
        """Test is_simple_graph returns False when self-loop exists"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'B')  # Self-loop
        
        assert graph.is_simple_graph() is False
    
    def test_is_simple_graph_empty(self):
        """Test is_simple_graph on empty graph"""
        graph = UndirectedGraph()
        assert graph.is_simple_graph() is True
    
    def test_get_degree_sequence(self):
        """Test getting degree sequence"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'C')
        
        sequence = graph.get_degree_sequence()
        
        # A has degree 2, B has degree 2, C has degree 2
        assert sequence == [2, 2, 2]
    
    def test_get_degree_sequence_star(self):
        """Test degree sequence of star graph"""
        graph = UndirectedGraph()
        center = 'Center'
        for i in range(5):
            graph.add_edge(center, f'Node{i}')
        
        sequence = graph.get_degree_sequence()
        # Center has degree 5, all others have degree 1
        assert sequence == [5, 1, 1, 1, 1, 1]  # Sorted descending
    
    def test_get_degree_sequence_sorted(self):
        """Test that degree sequence is properly sorted"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('A', 'D')
        graph.add_edge('B', 'C')
        
        sequence = graph.get_degree_sequence()
        # Should be sorted in descending order
        assert sequence == sorted(sequence, reverse=True)
    
    def test_get_graph_info_comprehensive(self):
        """Test comprehensive graph info"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        info = graph.get_graph_info()
        
        assert info['vertices'] == 3
        assert info['edges'] == 3
        assert info['is_simple'] is True
        assert info['min_degree'] == 2
        assert info['max_degree'] == 2
        assert info['average_degree'] == 2.0
        assert 'degree_sequence' in info
        assert 'vertex_degrees' in info
    
    def test_get_graph_info_empty(self):
        """Test graph info on empty graph"""
        graph = UndirectedGraph()
        info = graph.get_graph_info()
        
        assert info['vertices'] == 0
        assert info['edges'] == 0
        assert info['is_simple'] is True
        assert info['min_degree'] == 0
        assert info['max_degree'] == 0
        assert info['average_degree'] == 0.0
    
    def test_get_graph_info_with_self_loop(self):
        """Test graph info detects non-simple graph"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'B')  # Self-loop
        
        info = graph.get_graph_info()
        assert info['is_simple'] is False
    
    def test_get_graph_info_vertex_degrees(self):
        """Test vertex_degrees details in graph info"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        info = graph.get_graph_info()
        vertex_degrees = info['vertex_degrees']
        
        # Triangle graph - all vertices have degree 2
        assert vertex_degrees['A'] == 2
        assert vertex_degrees['B'] == 2
        assert vertex_degrees['C'] == 2
    
    def test_get_graph_info_star_graph(self):
        """Test graph info on star graph"""
        graph = UndirectedGraph()
        center = 'Center'
        for i in range(5):
            graph.add_edge(center, f'Node{i}')
        
        info = graph.get_graph_info()
        
        assert info['min_degree'] == 1  # Leaf nodes
        assert info['max_degree'] == 5  # Center node
        assert info['vertices'] == 6
        assert info['edges'] == 5
    
    def test_print_graph_analysis_runs(self):
        """Test that print_graph_analysis executes without errors"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        # Should not raise any exceptions
        graph.print_graph_analysis()
    
    def test_handshaking_lemma(self):
        """Test handshaking lemma: sum of degrees = 2 * edges"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        graph.add_edge('D', 'A')
        
        info = graph.get_graph_info()
        degree_sum = sum(info['vertex_degrees'].values())
        edge_count = info['edges']
        
        # Handshaking lemma
        assert degree_sum == 2 * edge_count
    
    def test_degree_sequence_graphical(self):
        """Test degree sequence properties"""
        graph = UndirectedGraph()
        # Create triangle
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        sequence = graph.get_degree_sequence()
        # Sum of degrees should be even (handshaking lemma)
        assert sum(sequence) % 2 == 0


class TestUndirectedGraphPaths:
    def test_find_path_exists(self):
        """Test finding a path between connected vertices"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        
        path = graph.find_path('A', 'D')
        assert path is not None
        assert path[0] == 'A'
        assert path[-1] == 'D'
        assert len(path) == 4
    
    def test_find_path_no_path(self):
        """Test finding path between disconnected vertices"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('C', 'D')
        
        path = graph.find_path('A', 'C')
        assert path is None
    
    def test_find_path_same_vertex(self):
        """Test path from vertex to itself"""
        graph = UndirectedGraph()
        graph.add_vertex('A')
        
        path = graph.find_path('A', 'A')
        assert path == ['A']
    
    def test_is_reachable_true(self):
        """Test reachability for connected vertices"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        assert graph.is_reachable('A', 'C') is True
    
    def test_is_reachable_false(self):
        """Test reachability for disconnected vertices"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_vertex('C')
        
        assert graph.is_reachable('A', 'C') is False
    
    def test_path_length(self):
        """Test calculating path length"""
        graph = UndirectedGraph()
        
        path = ['A', 'B', 'C', 'D']
        assert graph.path_length(path) == 3
        
        path = ['A']
        assert graph.path_length(path) == 0
        
        path = []
        assert graph.path_length(path) == 0
    
    def test_is_simple_path_true(self):
        """Test simple path (no repeated vertices)"""
        graph = UndirectedGraph()
        
        path = ['A', 'B', 'C', 'D']
        assert graph.is_simple_path(path) is True
    
    def test_is_simple_path_false(self):
        """Test non-simple path (repeated vertices)"""
        graph = UndirectedGraph()
        
        path = ['A', 'B', 'C', 'B', 'D']
        assert graph.is_simple_path(path) is False


class TestUndirectedGraphCycles:
    def test_has_cycle_true(self):
        """Test cycle detection in graph with cycle"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        assert graph.has_cycle() is True
    
    def test_has_cycle_false(self):
        """Test cycle detection in acyclic graph"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        assert graph.has_cycle() is False
    
    def test_find_cycles(self):
        """Test finding all cycles"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        cycles = graph.find_cycles()
        assert len(cycles) >= 1
    
    def test_is_acyclic_true(self):
        """Test acyclic property for tree"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'D')
        
        assert graph.is_acyclic() is True
    
    def test_is_acyclic_false(self):
        """Test acyclic property for graph with cycle"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        assert graph.is_acyclic() is False


class TestUndirectedGraphConnectivity:
    def test_is_connected_true(self):
        """Test connected graph"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        
        assert graph.is_connected() is True
    
    def test_is_connected_false(self):
        """Test disconnected graph"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('C', 'D')
        
        assert graph.is_connected() is False
    
    def test_is_connected_empty(self):
        """Test empty graph is connected"""
        graph = UndirectedGraph()
        assert graph.is_connected() is True
    
    def test_get_connected_components_single(self):
        """Test single connected component"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        components = graph.get_connected_components()
        assert len(components) == 1
        assert {'A', 'B', 'C'} == components[0]
    
    def test_get_connected_components_multiple(self):
        """Test multiple connected components"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('C', 'D')
        graph.add_vertex('E')
        
        components = graph.get_connected_components()
        assert len(components) == 3


class TestUndirectedGraphAdjacencyMatrix:
    def test_get_adjacency_matrix(self):
        """Test adjacency matrix generation"""
        graph = UndirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('A', 'C')
        
        matrix = graph.get_adjacency_matrix()
        assert len(matrix) == 3
        assert len(matrix[0]) == 3
        # Matrix should be symmetric
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                assert matrix[i][j] == matrix[j][i]
    
    def test_get_adjacency_matrix_empty(self):
        """Test adjacency matrix for empty graph"""
        graph = UndirectedGraph()
        matrix = graph.get_adjacency_matrix()
        assert matrix == []
    
    def test_from_adjacency_matrix(self):
        """Test creating graph from adjacency matrix"""
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        vertices = ['A', 'B', 'C']
        
        graph = UndirectedGraph.from_adjacency_matrix(matrix, vertices)
        assert graph.vertex_count() == 3
        assert graph.edge_count() == 3
        assert graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'C')
        assert graph.has_edge('A', 'C')
    
    def test_from_adjacency_matrix_default_vertices(self):
        """Test creating graph from matrix with default vertex labels"""
        matrix = [
            [0, 1],
            [1, 0]
        ]
        
        graph = UndirectedGraph.from_adjacency_matrix(matrix)
        assert graph.vertex_count() == 2
        assert graph.has_edge(0, 1)
    
    def test_adjacency_matrix_roundtrip(self):
        """Test converting to and from adjacency matrix"""
        original = UndirectedGraph()
        original.add_edge('A', 'B')
        original.add_edge('B', 'C')
        
        matrix = original.get_adjacency_matrix()
        reconstructed = UndirectedGraph.from_adjacency_matrix(matrix, ['A', 'B', 'C'])
        
        assert original.vertex_count() == reconstructed.vertex_count()
        assert original.edge_count() == reconstructed.edge_count()


