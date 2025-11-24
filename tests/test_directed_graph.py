import pytest
from pyhelper_jkluess.Complex.Graphs.directed_graph import DirectedGraph


class TestDirectedGraphCreation:
    def test_empty_graph_creation(self):
        """Test creating an empty directed graph"""
        graph = DirectedGraph()
        assert graph.vertex_count() == 0
        assert graph.edge_count() == 0
        assert graph.get_vertices() == []
        assert graph.get_edges() == []
    
    def test_graph_creation_with_data(self):
        """Test creating a graph with initial data"""
        data = {
            'A': ['B', 'C'],
            'B': ['C'],
            'C': []
        }
        graph = DirectedGraph(data)
        assert graph.vertex_count() == 3
        assert graph.edge_count() == 3


class TestVertexOperations:
    def test_add_single_vertex(self):
        """Test adding a single vertex"""
        graph = DirectedGraph()
        result = graph.add_vertex('A')
        assert result is True
        assert graph.has_vertex('A')
        assert graph.vertex_count() == 1
    
    def test_add_duplicate_vertex(self):
        """Test adding a duplicate vertex returns False"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        result = graph.add_vertex('A')
        assert result is False
        assert graph.vertex_count() == 1
    
    def test_add_multiple_vertices(self):
        """Test adding multiple vertices"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        graph.add_vertex('C')
        assert graph.vertex_count() == 3
        assert set(graph.get_vertices()) == {'A', 'B', 'C'}
    
    def test_remove_vertex(self):
        """Test removing a vertex"""
        graph = DirectedGraph()
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
        graph = DirectedGraph()
        result = graph.remove_vertex('Z')
        assert result is False
    
    def test_remove_vertex_removes_all_edges(self):
        """Test that removing a vertex removes all incoming and outgoing edges"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'A')
        graph.add_edge('C', 'A')
        
        graph.remove_vertex('A')
        assert not graph.has_edge('A', 'B')
        assert not graph.has_edge('B', 'A')
        assert graph.vertex_count() == 2


class TestEdgeOperations:
    def test_add_edge_between_existing_vertices(self):
        """Test adding an edge between existing vertices"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        result = graph.add_edge('A', 'B')
        
        assert result is True
        assert graph.has_edge('A', 'B')
        assert not graph.has_edge('B', 'A')  # Directed!
        assert graph.edge_count() == 1
    
    def test_add_edge_creates_vertices(self):
        """Test that adding edge creates vertices if they don't exist"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        
        assert graph.has_vertex('A')
        assert graph.has_vertex('B')
        assert graph.has_edge('A', 'B')
    
    def test_add_duplicate_edge(self):
        """Test adding a duplicate edge returns False"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        result = graph.add_edge('A', 'B')
        
        assert result is False
        assert graph.edge_count() == 1
    
    def test_add_reverse_edge(self):
        """Test adding reverse edge creates separate edge"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        result = graph.add_edge('B', 'A')
        
        assert result is True
        assert graph.edge_count() == 2
        assert graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'A')
    
    def test_remove_edge(self):
        """Test removing an edge"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        result = graph.remove_edge('A', 'B')
        
        assert result is True
        assert not graph.has_edge('A', 'B')
        assert graph.edge_count() == 0
    
    def test_remove_edge_directional(self):
        """Test removing edge only removes specified direction"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'A')
        graph.remove_edge('A', 'B')
        
        assert not graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'A')
        assert graph.edge_count() == 1
    
    def test_remove_nonexistent_edge(self):
        """Test removing an edge that doesn't exist"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        result = graph.remove_edge('A', 'B')
        
        assert result is False


class TestGraphQueries:
    def test_get_neighbors_empty(self):
        """Test getting neighbors of isolated vertex"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        assert graph.get_neighbors('A') == []
    
    def test_get_neighbors_outgoing(self):
        """Test getting outgoing neighbors (successors)"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'A')  # Incoming to A
        
        neighbors = graph.get_neighbors('A')
        assert set(neighbors) == {'B', 'C'}
    
    def test_get_predecessors_empty(self):
        """Test getting predecessors of isolated vertex"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        assert graph.get_predecessors('A') == []
    
    def test_get_predecessors(self):
        """Test getting predecessors (incoming neighbors)"""
        graph = DirectedGraph()
        graph.add_edge('B', 'A')
        graph.add_edge('C', 'A')
        graph.add_edge('A', 'D')  # Outgoing from A
        
        predecessors = graph.get_predecessors('A')
        assert set(predecessors) == {'B', 'C'}
    
    def test_get_neighbors_nonexistent_vertex(self):
        """Test getting neighbors of nonexistent vertex"""
        graph = DirectedGraph()
        assert graph.get_neighbors('Z') == []
    
    def test_get_predecessors_nonexistent_vertex(self):
        """Test getting predecessors of nonexistent vertex"""
        graph = DirectedGraph()
        assert graph.get_predecessors('Z') == []
    
    def test_get_vertices(self):
        """Test getting all vertices"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        graph.add_vertex('B')
        graph.add_vertex('C')
        
        vertices = graph.get_vertices()
        assert set(vertices) == {'A', 'B', 'C'}
    
    def test_get_edges(self):
        """Test getting all edges"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('A', 'C')
        
        edges = graph.get_edges()
        assert len(edges) == 3
        assert ('A', 'B') in edges
        assert ('B', 'C') in edges
        assert ('A', 'C') in edges
    
    def test_has_vertex(self):
        """Test checking vertex existence"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        
        assert graph.has_vertex('A')
        assert not graph.has_vertex('B')
    
    def test_has_edge(self):
        """Test checking edge existence"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        
        assert graph.has_edge('A', 'B')
        assert not graph.has_edge('B', 'A')  # Directed!
        assert not graph.has_edge('A', 'C')


class TestDegreeOperations:
    def test_in_degree_zero(self):
        """Test in-degree of vertex with no incoming edges"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        assert graph.in_degree('A') == 0
    
    def test_in_degree(self):
        """Test in-degree calculation"""
        graph = DirectedGraph()
        graph.add_edge('B', 'A')
        graph.add_edge('C', 'A')
        graph.add_edge('D', 'A')
        
        assert graph.in_degree('A') == 3
    
    def test_out_degree_zero(self):
        """Test out-degree of vertex with no outgoing edges"""
        graph = DirectedGraph()
        graph.add_vertex('A')
        assert graph.out_degree('A') == 0
    
    def test_out_degree(self):
        """Test out-degree calculation"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('A', 'D')
        
        assert graph.out_degree('A') == 3
    
    def test_degree_nonexistent_vertex(self):
        """Test degree of nonexistent vertex"""
        graph = DirectedGraph()
        assert graph.in_degree('Z') == 0
        assert graph.out_degree('Z') == 0
    
    def test_bidirectional_degrees(self):
        """Test degrees with bidirectional edges"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'A')
        
        assert graph.in_degree('A') == 1
        assert graph.out_degree('A') == 1
        assert graph.in_degree('B') == 1
        assert graph.out_degree('B') == 1


class TestGraphProperties:
    def test_vertex_count(self):
        """Test counting vertices"""
        graph = DirectedGraph()
        assert graph.vertex_count() == 0
        
        graph.add_vertex('A')
        assert graph.vertex_count() == 1
        
        graph.add_vertex('B')
        graph.add_vertex('C')
        assert graph.vertex_count() == 3
    
    def test_edge_count(self):
        """Test counting edges"""
        graph = DirectedGraph()
        assert graph.edge_count() == 0
        
        graph.add_edge('A', 'B')
        assert graph.edge_count() == 1
        
        graph.add_edge('B', 'A')  # Reverse edge
        assert graph.edge_count() == 2


class TestGraphWithDifferentDataTypes:
    def test_integer_vertices(self):
        """Test graph with integer vertices"""
        graph = DirectedGraph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        
        assert graph.has_vertex(1)
        assert graph.has_edge(1, 2)
        assert graph.get_neighbors(2) == [3]
    
    def test_mixed_type_vertices(self):
        """Test graph with mixed type vertices"""
        graph = DirectedGraph()
        graph.add_edge('A', 1)
        graph.add_edge(1, 2.5)
        
        assert graph.has_vertex('A')
        assert graph.has_vertex(1)
        assert graph.has_vertex(2.5)
        assert graph.has_edge('A', 1)


class TestGraphStringRepresentation:
    def test_str_empty_graph(self):
        """Test string representation of empty graph"""
        graph = DirectedGraph()
        assert str(graph) == "Empty graph"
    
    def test_str_graph_with_data(self):
        """Test string representation of graph with data"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        result = str(graph)
        
        assert "Directed Graph:" in result
        assert "A" in result
    
    def test_repr_graph(self):
        """Test repr of graph"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        result = repr(graph)
        assert "DirectedGraph" in result
        assert "vertices=3" in result
        assert "edges=2" in result


class TestComplexGraphScenarios:
    def test_cycle_detection_simple(self):
        """Test graph with simple cycle"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        assert graph.vertex_count() == 3
        assert graph.edge_count() == 3
        # A cycle exists: A -> B -> C -> A
    
    def test_dag_directed_acyclic_graph(self):
        """Test directed acyclic graph (DAG)"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'D')
        graph.add_edge('C', 'D')
        
        assert graph.vertex_count() == 4
        assert graph.edge_count() == 4
        # No cycles - could be topologically sorted
    
    def test_strongly_connected_component(self):
        """Test strongly connected component"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        # All vertices can reach each other
        # A -> B -> C -> A
        assert graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'C')
        assert graph.has_edge('C', 'A')
    
    def test_star_graph_directed(self):
        """Test directed star graph"""
        graph = DirectedGraph()
        center = 'Center'
        
        # All edges point away from center
        for i in range(5):
            graph.add_edge(center, f'Node{i}')
        
        assert graph.out_degree(center) == 5
        assert graph.in_degree(center) == 0
    
    def test_reverse_star_graph(self):
        """Test reverse star graph (all point to center)"""
        graph = DirectedGraph()
        center = 'Center'
        
        # All edges point to center
        for i in range(5):
            graph.add_edge(f'Node{i}', center)
        
        assert graph.out_degree(center) == 0
        assert graph.in_degree(center) == 5
    
    def test_self_loop(self):
        """Test self-loop (edge from vertex to itself)"""
        graph = DirectedGraph()
        graph.add_edge('A', 'A')
        
        assert graph.has_vertex('A')
        assert graph.has_edge('A', 'A')
        assert graph.in_degree('A') == 1
        assert graph.out_degree('A') == 1
    
    def test_parallel_paths(self):
        """Test graph with multiple paths between vertices"""
        graph = DirectedGraph()
        # Path 1: A -> B -> D
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'D')
        # Path 2: A -> C -> D
        graph.add_edge('A', 'C')
        graph.add_edge('C', 'D')
        
        assert graph.out_degree('A') == 2
        assert graph.in_degree('D') == 2
        assert set(graph.get_neighbors('A')) == {'B', 'C'}
        assert set(graph.get_predecessors('D')) == {'B', 'C'}


class TestDirectedGraphTheoryMethods:
    """Test new graph theory analysis methods"""
    
    def test_degree_calculation(self):
        """Test total degree calculation (in + out)"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        # Each vertex has in-degree 1 and out-degree 1
        assert graph.degree('A') == 2
        assert graph.degree('B') == 2
        assert graph.degree('C') == 2
    
    def test_degree_asymmetric(self):
        """Test degree with asymmetric in/out degrees"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('A', 'D')
        
        assert graph.out_degree('A') == 3
        assert graph.in_degree('A') == 0
        assert graph.degree('A') == 3
        
        assert graph.in_degree('B') == 1
        assert graph.out_degree('B') == 0
        assert graph.degree('B') == 1
    
    def test_degree_nonexistent_vertex(self):
        """Test degree of non-existent vertex"""
        graph = DirectedGraph()
        assert graph.degree('Z') == 0
    
    def test_is_simple_graph_true(self):
        """Test is_simple_graph returns True for graph without self-loops"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        assert graph.is_simple_graph() is True
    
    def test_is_simple_graph_false(self):
        """Test is_simple_graph returns False when self-loop exists"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'B')  # Self-loop
        
        assert graph.is_simple_graph() is False
    
    def test_is_simple_graph_multiple_self_loops(self):
        """Test is_simple_graph with multiple self-loops"""
        graph = DirectedGraph()
        graph.add_edge('A', 'A')
        graph.add_edge('B', 'B')
        
        assert graph.is_simple_graph() is False
    
    def test_is_simple_graph_empty(self):
        """Test is_simple_graph on empty graph"""
        graph = DirectedGraph()
        assert graph.is_simple_graph() is True
    
    def test_get_degree_sequence(self):
        """Test getting degree sequences"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'C')
        
        sequences = graph.get_degree_sequence()
        
        assert 'in_degrees' in sequences
        assert 'out_degrees' in sequences
        assert 'total_degrees' in sequences
        
        # A: out=2, in=0, total=2
        # B: out=1, in=1, total=2
        # C: out=0, in=2, total=2
        assert sequences['in_degrees'] == [2, 1, 0]  # Sorted descending
        assert sequences['out_degrees'] == [2, 1, 0]
        assert sequences['total_degrees'] == [2, 2, 2]
    
    def test_get_degree_sequence_cycle(self):
        """Test degree sequence on cycle graph"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        sequences = graph.get_degree_sequence()
        
        # All vertices have in=1, out=1, total=2
        assert sequences['in_degrees'] == [1, 1, 1]
        assert sequences['out_degrees'] == [1, 1, 1]
        assert sequences['total_degrees'] == [2, 2, 2]
    
    def test_get_graph_info_comprehensive(self):
        """Test comprehensive graph info"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'C')
        
        info = graph.get_graph_info()
        
        assert info['vertices'] == 3
        assert info['edges'] == 3
        assert info['is_simple'] is True
        assert info['min_in_degree'] == 0  # A has no incoming
        assert info['max_in_degree'] == 2  # C has 2 incoming
        assert info['min_out_degree'] == 0  # C has no outgoing
        assert info['max_out_degree'] == 2  # A has 2 outgoing
        assert 'average_in_degree' in info
        assert 'average_out_degree' in info
        assert 'vertex_degrees' in info
    
    def test_get_graph_info_empty(self):
        """Test graph info on empty graph"""
        graph = DirectedGraph()
        info = graph.get_graph_info()
        
        assert info['vertices'] == 0
        assert info['edges'] == 0
        assert info['is_simple'] is True
        assert info['min_in_degree'] == 0
        assert info['max_in_degree'] == 0
    
    def test_get_graph_info_with_self_loop(self):
        """Test graph info detects non-simple graph"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'B')  # Self-loop
        
        info = graph.get_graph_info()
        assert info['is_simple'] is False
    
    def test_get_graph_info_vertex_degrees(self):
        """Test vertex_degrees details in graph info"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        info = graph.get_graph_info()
        vertex_degrees = info['vertex_degrees']
        
        assert 'A' in vertex_degrees
        assert vertex_degrees['A']['in_degree'] == 0
        assert vertex_degrees['A']['out_degree'] == 1
        assert vertex_degrees['A']['total_degree'] == 1
        
        assert vertex_degrees['B']['in_degree'] == 1
        assert vertex_degrees['B']['out_degree'] == 1
        assert vertex_degrees['B']['total_degree'] == 2
        
        assert vertex_degrees['C']['in_degree'] == 1
        assert vertex_degrees['C']['out_degree'] == 0
        assert vertex_degrees['C']['total_degree'] == 1
    
    def test_print_graph_analysis_runs(self):
        """Test that print_graph_analysis executes without errors"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        # Should not raise any exceptions
        graph.print_graph_analysis()
    
    def test_degree_sequence_sorting(self):
        """Test that degree sequences are properly sorted"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('A', 'D')
        graph.add_edge('B', 'C')
        
        sequences = graph.get_degree_sequence()
        
        # Should be sorted in descending order
        assert sequences['in_degrees'] == sorted(sequences['in_degrees'], reverse=True)
        assert sequences['out_degrees'] == sorted(sequences['out_degrees'], reverse=True)
        assert sequences['total_degrees'] == sorted(sequences['total_degrees'], reverse=True)


class TestDirectedGraphPaths:
    def test_find_path_exists(self):
        """Test finding a directed path"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        
        path = graph.find_path('A', 'D')
        assert path is not None
        assert path[0] == 'A'
        assert path[-1] == 'D'
    
    def test_find_path_no_path_wrong_direction(self):
        """Test path doesn't exist against edge direction"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        path = graph.find_path('C', 'A')
        assert path is None
    
    def test_find_path_disconnected(self):
        """Test finding path between disconnected vertices"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('C', 'D')
        
        path = graph.find_path('A', 'C')
        assert path is None
    
    def test_is_reachable_true(self):
        """Test reachability following directed edges"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        assert graph.is_reachable('A', 'C') is True
        assert graph.is_reachable('C', 'A') is False
    
    def test_path_length(self):
        """Test calculating path length"""
        graph = DirectedGraph()
        
        path = ['A', 'B', 'C']
        assert graph.path_length(path) == 2
    
    def test_is_simple_path(self):
        """Test simple path detection"""
        graph = DirectedGraph()
        
        assert graph.is_simple_path(['A', 'B', 'C']) is True
        assert graph.is_simple_path(['A', 'B', 'A']) is False


class TestDirectedGraphCycles:
    def test_has_cycle_true(self):
        """Test cycle detection in directed graph"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        assert graph.has_cycle() is True
    
    def test_has_cycle_false_dag(self):
        """Test DAG (no cycles)"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'D')
        graph.add_edge('C', 'D')
        
        assert graph.has_cycle() is False
    
    def test_find_cycles(self):
        """Test finding cycles"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        cycles = graph.find_cycles()
        assert len(cycles) >= 1
    
    def test_is_acyclic_dag(self):
        """Test DAG is acyclic"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        assert graph.is_acyclic() is True
    
    def test_is_acyclic_with_cycle(self):
        """Test graph with cycle is not acyclic"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'A')
        
        assert graph.is_acyclic() is False


class TestDirectedGraphConnectivity:
    def test_is_strongly_connected_true(self):
        """Test strongly connected graph"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        assert graph.is_strongly_connected() is True
    
    def test_is_strongly_connected_false(self):
        """Test not strongly connected graph"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        assert graph.is_strongly_connected() is False
    
    def test_get_strongly_connected_components_single(self):
        """Test single strongly connected component"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'A')
        
        components = graph.get_strongly_connected_components()
        assert len(components) == 1
        assert {'A', 'B', 'C'} == components[0]
    
    def test_get_strongly_connected_components_multiple(self):
        """Test multiple strongly connected components"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'A')
        graph.add_edge('C', 'D')
        graph.add_edge('D', 'C')
        
        components = graph.get_strongly_connected_components()
        assert len(components) == 2


class TestDirectedGraphAdjacencyMatrix:
    def test_get_adjacency_matrix(self):
        """Test adjacency matrix for directed graph"""
        graph = DirectedGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        matrix = graph.get_adjacency_matrix()
        assert len(matrix) == 3
        # Matrix should NOT be symmetric for directed graphs
    
    def test_from_adjacency_matrix(self):
        """Test creating directed graph from adjacency matrix"""
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        vertices = ['A', 'B', 'C']
        
        graph = DirectedGraph.from_adjacency_matrix(matrix, vertices)
        assert graph.vertex_count() == 3
        assert graph.has_edge('A', 'B')
        assert graph.has_edge('B', 'C')
        assert graph.has_edge('C', 'A')
        assert not graph.has_edge('B', 'A')  # Direction matters
    
    def test_adjacency_matrix_roundtrip(self):
        """Test converting to and from adjacency matrix"""
        original = DirectedGraph()
        original.add_edge('A', 'B')
        original.add_edge('B', 'C')
        
        matrix = original.get_adjacency_matrix()
        reconstructed = DirectedGraph.from_adjacency_matrix(matrix, ['A', 'B', 'C'])
        
        assert original.vertex_count() == reconstructed.vertex_count()
        assert original.edge_count() == reconstructed.edge_count()


