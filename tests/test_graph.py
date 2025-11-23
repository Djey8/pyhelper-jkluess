import pytest
from Complex.Graphs.graph import Graph


class TestGraphCreation:
    def test_empty_undirected_unweighted(self):
        """Test creating an empty undirected unweighted graph"""
        g = Graph(directed=False, weighted=False)
        assert g.vertex_count() == 0
        assert g.edge_count() == 0
        assert not g.is_directed
        assert not g.is_weighted
    
    def test_empty_directed_unweighted(self):
        """Test creating an empty directed unweighted graph"""
        g = Graph(directed=True, weighted=False)
        assert g.vertex_count() == 0
        assert not g.is_weighted
        assert g.is_directed
    
    def test_empty_undirected_weighted(self):
        """Test creating an empty undirected weighted graph"""
        g = Graph(directed=False, weighted=True)
        assert g.is_weighted
        assert not g.is_directed
    
    def test_empty_directed_weighted(self):
        """Test creating an empty directed weighted graph"""
        g = Graph(directed=True, weighted=True)
        assert g.is_weighted
        assert g.is_directed
    
    def test_undirected_unweighted_with_data(self):
        """Test creating undirected unweighted graph with data"""
        data = {'A': ['B', 'C'], 'B': ['C']}
        g = Graph(directed=False, weighted=False, data=data)
        assert g.vertex_count() == 3
        assert g.edge_count() == 3  # A-B, A-C, B-C
        assert g.has_edge('A', 'B')
        assert g.has_edge('B', 'A')  # Symmetric
    
    def test_directed_unweighted_with_data(self):
        """Test creating directed unweighted graph with data"""
        data = {'A': ['B'], 'B': ['C']}
        g = Graph(directed=True, weighted=False, data=data)
        assert g.vertex_count() == 3
        assert g.edge_count() == 2
        assert g.has_edge('A', 'B')
        assert not g.has_edge('B', 'A')  # Not symmetric
    
    def test_undirected_weighted_with_data(self):
        """Test creating undirected weighted graph with data"""
        data = {'A': [('B', 5), ('C', 10)], 'B': [('C', 3)]}
        g = Graph(directed=False, weighted=True, data=data)
        assert g.vertex_count() == 3
        assert g.get_edge_weight('A', 'B') == 5
        assert g.get_edge_weight('B', 'A') == 5  # Symmetric
    
    def test_directed_weighted_with_data(self):
        """Test creating directed weighted graph with data"""
        data = {'A': [('B', 10)], 'B': [('C', 20)]}
        g = Graph(directed=True, weighted=True, data=data)
        assert g.get_edge_weight('A', 'B') == 10
        assert g.get_edge_weight('B', 'A') is None  # Not symmetric


class TestVertexOperations:
    def test_add_vertex_undirected(self):
        """Test adding vertices to undirected graph"""
        g = Graph(directed=False, weighted=False)
        assert g.add_vertex('A') is True
        assert g.add_vertex('A') is False  # Duplicate
        assert g.has_vertex('A')
        assert g.vertex_count() == 1
    
    def test_add_vertex_directed(self):
        """Test adding vertices to directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_vertex('X')
        assert g.has_vertex('X')
    
    def test_remove_vertex_undirected(self):
        """Test removing vertex from undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        assert g.remove_vertex('A') is True
        assert not g.has_vertex('A')
        assert not g.has_edge('A', 'B')
    
    def test_remove_vertex_directed(self):
        """Test removing vertex from directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.remove_vertex('A')
        assert not g.has_edge('A', 'B')
    
    def test_get_vertices(self):
        """Test getting all vertices"""
        g = Graph(directed=False, weighted=False)
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        assert set(g.get_vertices()) == {'A', 'B', 'C'}


class TestEdgeOperations:
    def test_add_edge_undirected_unweighted(self):
        """Test adding edge to undirected unweighted graph"""
        g = Graph(directed=False, weighted=False)
        assert g.add_edge('A', 'B') is True
        assert g.has_edge('A', 'B')
        assert g.has_edge('B', 'A')  # Symmetric
        assert g.edge_count() == 1
    
    def test_add_edge_directed_unweighted(self):
        """Test adding edge to directed unweighted graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        assert g.has_edge('A', 'B')
        assert not g.has_edge('B', 'A')  # Not symmetric
    
    def test_add_edge_undirected_weighted(self):
        """Test adding weighted edge to undirected graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 10)
        assert g.get_edge_weight('A', 'B') == 10
        assert g.get_edge_weight('B', 'A') == 10  # Symmetric
    
    def test_add_edge_directed_weighted(self):
        """Test adding weighted edge to directed graph"""
        g = Graph(directed=True, weighted=True)
        g.add_edge('A', 'B', 15)
        assert g.get_edge_weight('A', 'B') == 15
        assert g.get_edge_weight('B', 'A') is None
    
    def test_remove_edge_undirected(self):
        """Test removing edge from undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        assert g.remove_edge('A', 'B') is True
        assert not g.has_edge('A', 'B')
        assert not g.has_edge('B', 'A')
    
    def test_remove_edge_directed(self):
        """Test removing edge from directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.remove_edge('A', 'B')
        assert not g.has_edge('A', 'B')
    
    def test_update_edge_weight_weighted(self):
        """Test updating edge weight in weighted graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 10)
        assert g.update_edge_weight('A', 'B', 20) is True
        assert g.get_edge_weight('A', 'B') == 20
    
    def test_update_edge_weight_unweighted_fails(self):
        """Test that updating weight fails for unweighted graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        assert g.update_edge_weight('A', 'B', 10) is False
    
    def test_get_edge_weight_unweighted_returns_none(self):
        """Test that getting weight from unweighted graph returns None"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        assert g.get_edge_weight('A', 'B') is None


class TestNeighbors:
    def test_get_neighbors_undirected(self):
        """Test getting neighbors in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        neighbors = g.get_neighbors('A')
        assert set(neighbors) == {'B', 'C'}
    
    def test_get_neighbors_directed(self):
        """Test getting neighbors (successors) in directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('C', 'A')
        assert g.get_neighbors('A') == ['B']
    
    def test_get_weighted_neighbors(self):
        """Test getting weighted neighbors"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 5)
        g.add_edge('A', 'C', 10)
        neighbors = g.get_weighted_neighbors('A')
        assert len(neighbors) == 2
        assert ('B', 5) in neighbors
        assert ('C', 10) in neighbors
    
    def test_get_predecessors_directed(self):
        """Test getting predecessors in directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'C')
        g.add_edge('B', 'C')
        predecessors = g.get_predecessors('C')
        assert set(predecessors) == {'A', 'B'}
    
    def test_get_predecessors_undirected_same_as_neighbors(self):
        """Test that predecessors equal neighbors in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        assert set(g.get_predecessors('A')) == set(g.get_neighbors('A'))


class TestDegree:
    def test_degree_undirected(self):
        """Test degree in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        assert g.degree('A') == 2
        assert g.degree('B') == 1
    
    def test_in_out_degree_directed(self):
        """Test in-degree and out-degree in directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('D', 'A')
        assert g.out_degree('A') == 2
        assert g.in_degree('A') == 1
        assert g.degree('A') == 3  # in + out
    
    def test_weighted_degree_undirected(self):
        """Test weighted degree in undirected weighted graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 5)
        g.add_edge('A', 'C', 10)
        assert g.weighted_degree('A') == 15
    
    def test_weighted_degree_directed(self):
        """Test weighted in/out degree in directed weighted graph"""
        g = Graph(directed=True, weighted=True)
        g.add_edge('A', 'B', 10)
        g.add_edge('A', 'C', 20)
        g.add_edge('D', 'A', 5)
        assert g.weighted_out_degree('A') == 30
        assert g.weighted_in_degree('A') == 5
        assert g.weighted_degree('A') == 35


class TestPaths:
    def test_find_path_undirected(self):
        """Test finding path in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        path = g.find_path('A', 'C')
        assert path == ['A', 'B', 'C']
    
    def test_find_path_directed(self):
        """Test finding path in directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        assert g.find_path('A', 'C') == ['A', 'B', 'C']
        assert g.find_path('C', 'A') is None  # No reverse path
    
    def test_is_reachable(self):
        """Test reachability check"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        assert g.is_reachable('A', 'C') is True
        assert g.is_reachable('C', 'A') is False
    
    def test_path_length(self):
        """Test path length calculation"""
        g = Graph(directed=False, weighted=False)
        path = ['A', 'B', 'C', 'D']
        assert g.path_length(path) == 3
    
    def test_path_weight_weighted(self):
        """Test path weight in weighted graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 5)
        g.add_edge('B', 'C', 10)
        path = ['A', 'B', 'C']
        assert g.path_weight(path) == 15
    
    def test_path_weight_unweighted(self):
        """Test that path weight returns edge count for unweighted graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        path = ['A', 'B', 'C']
        assert g.path_weight(path) == 2
    
    def test_is_simple_path(self):
        """Test simple path check"""
        g = Graph(directed=False, weighted=False)
        assert g.is_simple_path(['A', 'B', 'C']) is True
        assert g.is_simple_path(['A', 'B', 'A']) is False


class TestCycles:
    def test_has_cycle_undirected(self):
        """Test cycle detection in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        assert g.has_cycle() is False
        
        g.add_edge('C', 'A')
        assert g.has_cycle() is True
    
    def test_has_cycle_directed(self):
        """Test cycle detection in directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        assert g.has_cycle() is False
        
        g.add_edge('C', 'A')
        assert g.has_cycle() is True
    
    def test_find_cycles_undirected(self):
        """Test finding cycles in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        cycles = g.find_cycles()
        assert len(cycles) > 0
    
    def test_find_cycles_directed(self):
        """Test finding cycles in directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        cycles = g.find_cycles()
        assert len(cycles) > 0
    
    def test_is_acyclic(self):
        """Test acyclic check"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        assert g.is_acyclic() is True
        
        g.add_edge('C', 'A')
        assert g.is_acyclic() is False


class TestConnectivity:
    def test_is_connected_undirected(self):
        """Test connectivity in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        assert g.is_connected() is True
        
        g.add_vertex('D')
        assert g.is_connected() is False
    
    def test_get_connected_components_undirected(self):
        """Test finding connected components in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('C', 'D')
        components = g.get_connected_components()
        assert len(components) == 2
        assert {'A', 'B'} in components
        assert {'C', 'D'} in components
    
    def test_is_strongly_connected_directed(self):
        """Test strong connectivity in directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        assert g.is_strongly_connected() is True
    
    def test_get_strongly_connected_components(self):
        """Test finding strongly connected components"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'A')
        g.add_edge('C', 'D')
        g.add_edge('D', 'C')
        components = g.get_strongly_connected_components()
        assert len(components) == 2


class TestAdjacencyMatrix:
    def test_get_adjacency_matrix_undirected_unweighted(self):
        """Test getting adjacency matrix for undirected unweighted graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        matrix = g.get_adjacency_matrix()
        # A-B-C: matrix should be symmetric with 1s for edges
        assert matrix[0][1] == 1  # A-B
        assert matrix[1][0] == 1  # B-A
        assert matrix[1][2] == 1  # B-C
        assert matrix[2][1] == 1  # C-B
    
    def test_get_adjacency_matrix_directed_unweighted(self):
        """Test getting adjacency matrix for directed unweighted graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        matrix = g.get_adjacency_matrix()
        assert matrix[0][1] == 1  # A->B
        assert matrix[1][0] == 0  # Not symmetric
    
    def test_get_adjacency_matrix_weighted(self):
        """Test getting adjacency matrix for weighted graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 10)
        g.add_edge('B', 'C', 20)
        matrix = g.get_adjacency_matrix()
        assert matrix[0][1] == 10
        assert matrix[1][0] == 10  # Symmetric
        assert matrix[1][2] == 20
    
    def test_from_adjacency_matrix_undirected(self):
        """Test creating graph from adjacency matrix"""
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        g = Graph.from_adjacency_matrix(matrix, ['A', 'B', 'C'], directed=False, weighted=False)
        assert g.has_edge('A', 'B')
        assert g.has_edge('B', 'C')
        assert g.edge_count() == 3
    
    def test_from_adjacency_matrix_directed(self):
        """Test creating directed graph from adjacency matrix"""
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ]
        g = Graph.from_adjacency_matrix(matrix, ['A', 'B', 'C'], directed=True, weighted=False)
        assert g.has_edge('A', 'B')
        assert g.has_edge('B', 'C')
        assert not g.has_edge('B', 'A')
    
    def test_from_adjacency_matrix_weighted(self):
        """Test creating weighted graph from adjacency matrix"""
        matrix = [
            [0, 10, 0],
            [10, 0, 20],
            [0, 20, 0]
        ]
        g = Graph.from_adjacency_matrix(matrix, ['A', 'B', 'C'], directed=False, weighted=True)
        assert g.get_edge_weight('A', 'B') == 10
        assert g.get_edge_weight('B', 'C') == 20


class TestGraphProperties:
    def test_is_simple_graph(self):
        """Test simple graph check (no self-loops)"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        assert g.is_simple_graph() is True
        
        g.add_edge('A', 'A')  # Self-loop
        assert g.is_simple_graph() is False
    
    def test_vertex_count(self):
        """Test vertex count"""
        g = Graph(directed=False, weighted=False)
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_vertex('C')
        assert g.vertex_count() == 3
    
    def test_edge_count_undirected(self):
        """Test edge count in undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        assert g.edge_count() == 2
    
    def test_edge_count_directed(self):
        """Test edge count in directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'A')
        assert g.edge_count() == 2
    
    def test_total_weight(self):
        """Test total weight calculation"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 10)
        g.add_edge('B', 'C', 20)
        assert g.total_weight() == 30
    
    def test_get_edges_undirected(self):
        """Test getting edges from undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        edges = g.get_edges()
        assert len(edges) == 2
        assert ('A', 'B') in edges or ('B', 'A') in edges
    
    def test_get_edges_directed(self):
        """Test getting edges from directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'A')
        edges = g.get_edges()
        assert len(edges) == 2
        assert ('A', 'B') in edges
        assert ('B', 'A') in edges
    
    def test_get_edges_weighted(self):
        """Test getting edges from weighted graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 10)
        edges = g.get_edges()
        assert len(edges) == 1
        assert edges[0][2] == 10  # Third element is weight


class TestStringRepresentation:
    def test_str_undirected_unweighted(self):
        """Test string representation of undirected unweighted graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        s = str(g)
        assert 'Undirected' in s
        assert 'Graph' in s
    
    def test_str_directed_weighted(self):
        """Test string representation of directed weighted graph"""
        g = Graph(directed=True, weighted=True)
        g.add_edge('A', 'B', 10)
        s = str(g)
        assert 'Directed' in s
        assert 'Weighted' in s
    
    def test_repr(self):
        """Test repr of graph"""
        g = Graph(directed=True, weighted=True)
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_edge('A', 'B', 5)
        r = repr(g)
        assert 'Graph' in r
        assert 'Weighted' in r
        assert 'Directed' in r


class TestMixedModes:
    def test_all_four_modes_coexist(self):
        """Test that all four graph modes can coexist"""
        g1 = Graph(directed=False, weighted=False)
        g2 = Graph(directed=True, weighted=False)
        g3 = Graph(directed=False, weighted=True)
        g4 = Graph(directed=True, weighted=True)
        
        g1.add_edge('A', 'B')
        g2.add_edge('A', 'B')
        g3.add_edge('A', 'B', 10)
        g4.add_edge('A', 'B', 20)
        
        assert g1.has_edge('B', 'A')  # Undirected
        assert not g2.has_edge('B', 'A')  # Directed
        assert g3.get_edge_weight('A', 'B') == 10
        assert g4.get_edge_weight('A', 'B') == 20
        assert g4.get_edge_weight('B', 'A') is None  # Directed
