import pytest
from pyhelper_jkluess.Complex.Graphs.graph import Graph


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


class TestGetAdjacencyList:
    def test_get_adjacency_list_unweighted_undirected(self):
        """Test getting adjacency list from unweighted undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        
        adj_list = g.get_adjacency_list()
        assert 'A' in adj_list
        assert 'B' in adj_list['A']
        assert 'A' in adj_list['B']  # Symmetric
        assert 'C' in adj_list['B']
    
    def test_get_adjacency_list_weighted_undirected(self):
        """Test getting adjacency list from weighted undirected graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 10)
        g.add_edge('B', 'C', 5)
        
        adj_list = g.get_adjacency_list()
        # For weighted graphs, adjacency list is Dict[vertex, List[(neighbor, weight)]]
        assert isinstance(adj_list['A'], list)
        assert ('B', 10) in adj_list['A']
        assert ('A', 10) in adj_list['B']  # Symmetric
        assert ('C', 5) in adj_list['B']
        assert ('B', 5) in adj_list['C']
    
    def test_get_adjacency_list_directed(self):
        """Test getting adjacency list from directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        
        adj_list = g.get_adjacency_list()
        assert 'B' in adj_list['A']
        assert 'A' not in adj_list['B']  # Not symmetric
        assert 'C' in adj_list['B']
    
    def test_create_graph_from_adjacency_list(self):
        """Test creating a new graph from adjacency list"""
        g1 = Graph(directed=False, weighted=True)
        g1.add_edge('A', 'B', 10)
        g1.add_edge('B', 'C', 5)
        g1.add_edge('A', 'C', 8)
        
        adj_list = g1.get_adjacency_list()
        g2 = Graph(directed=False, weighted=True, data=adj_list)
        
        assert g2.vertex_count() == g1.vertex_count()
        assert g2.edge_count() == g1.edge_count()
        assert g2.get_edge_weight('A', 'B') == 10
        assert g2.get_edge_weight('B', 'C') == 5
        assert g2.get_edge_weight('A', 'C') == 8
    
    def test_adjacency_list_is_deep_copy(self):
        """Test that adjacency list is a deep copy"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 10)
        
        adj_list = g.get_adjacency_list()
        # Modify the copy
        adj_list['A'].append(('C', 999))
        
        assert g.get_edge_weight('A', 'B') == 10  # Original unchanged
        assert not g.has_edge('A', 'C')  # New edge not in original
    
    def test_get_adjacency_list_empty_graph(self):
        """Test getting adjacency list from empty graph"""
        g = Graph(directed=False, weighted=False)
        adj_list = g.get_adjacency_list()
        assert adj_list == {}


class TestFindShortestPath:
    def test_shortest_path_unweighted_undirected(self):
        """Test shortest path in unweighted undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('A', 'D')
        g.add_edge('D', 'C')
        
        path, distance = g.find_shortest_path('A', 'C')
        assert len(path) == 3  # Either A-B-C or A-D-C
        assert distance == 2
        assert path[0] == 'A'
        assert path[-1] == 'C'
    
    def test_shortest_path_weighted_undirected(self):
        """Test shortest path in weighted undirected graph using Dijkstra"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 4)
        g.add_edge('A', 'C', 2)
        g.add_edge('C', 'B', 1)
        g.add_edge('B', 'D', 5)
        g.add_edge('C', 'D', 8)
        
        path, distance = g.find_shortest_path('A', 'D')
        assert path == ['A', 'C', 'B', 'D']
        assert distance == 8  # 2 + 1 + 5
    
    def test_shortest_path_weighted_directed(self):
        """Test shortest path in weighted directed graph"""
        g = Graph(directed=True, weighted=True)
        g.add_edge('A', 'B', 1)
        g.add_edge('A', 'C', 4)
        g.add_edge('B', 'C', 2)
        g.add_edge('C', 'D', 1)
        g.add_edge('B', 'D', 5)
        
        path, distance = g.find_shortest_path('A', 'D')
        assert path == ['A', 'B', 'C', 'D']
        assert distance == 4  # 1 + 2 + 1
    
    def test_shortest_path_same_vertex(self):
        """Test shortest path when start equals end"""
        g = Graph(directed=False, weighted=False)
        g.add_vertex('A')
        
        path, distance = g.find_shortest_path('A', 'A')
        assert path == ['A']
        assert distance == 0
    
    def test_shortest_path_no_path_exists(self):
        """Test shortest path when no path exists"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('C', 'D')
        
        result = g.find_shortest_path('A', 'D')
        assert result is None
    
    def test_shortest_path_nonexistent_vertex(self):
        """Test shortest path with nonexistent vertex"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        
        result = g.find_shortest_path('A', 'Z')
        assert result is None
        
        result = g.find_shortest_path('Z', 'A')
        assert result is None
    
    def test_shortest_path_unweighted_directed(self):
        """Test shortest path in unweighted directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        g.add_edge('C', 'D')
        g.add_edge('C', 'E')
        g.add_edge('D', 'E')
        
        path, distance = g.find_shortest_path('A', 'E')
        # Shortest path is A->C->E (2 edges)
        assert distance == 2
        assert path == ['A', 'C', 'E']
        assert path[0] == 'A'
        assert path[-1] == 'E'
    
    def test_shortest_path_weighted_multiple_paths(self):
        """Test shortest path when multiple paths exist with different weights"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 1)
        g.add_edge('B', 'C', 1)
        g.add_edge('A', 'C', 10)  # Direct but expensive
        
        path, distance = g.find_shortest_path('A', 'C')
        assert path == ['A', 'B', 'C']
        assert distance == 2  # 1 + 1
    
    def test_shortest_path_weighted_negative_weights(self):
        """Test shortest path handles graphs with negative weights (Dijkstra limitation)"""
        # Note: Dijkstra doesn't work correctly with negative weights
        # This test documents the behavior
        g = Graph(directed=True, weighted=True)
        g.add_edge('A', 'B', 5)
        g.add_edge('A', 'C', 2)
        g.add_edge('C', 'B', -10)  # Negative weight
        
        # Dijkstra will find a path, but may not be optimal with negative weights
        path, distance = g.find_shortest_path('A', 'B')
        assert path is not None
        assert path[0] == 'A'
        assert path[-1] == 'B'
    
    def test_shortest_path_complex_graph(self):
        """Test shortest path in a complex weighted graph"""
        g = Graph(directed=False, weighted=True)
        edges = [
            ('A', 'B', 7), ('A', 'C', 9), ('A', 'F', 14),
            ('B', 'C', 10), ('B', 'D', 15), ('C', 'D', 11),
            ('C', 'F', 2), ('D', 'E', 6), ('E', 'F', 9)
        ]
        for u, v, w in edges:
            g.add_edge(u, v, w)
        
        path, distance = g.find_shortest_path('A', 'E')
        assert path == ['A', 'C', 'F', 'E'] or path == ['A', 'C', 'D', 'E']
        assert distance == 20  # A->C(9) + C->F(2) + F->E(9) or A->C(9) + C->D(11) + D->E(6)...
        # Actually: A->C(9) + C->D(11) + D->E(6) = 26 vs A->C(9) + C->F(2) + F->E(9) = 20
        assert distance == 20


class TestDFS:
    """Test cases for depth-first search (DFS) traversal"""
    
    def test_dfs_undirected_simple(self):
        """Test DFS traversal on simple undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('A', 'D')
        
        result = g.dfs('A')
        assert 'A' == result[0]  # Start vertex is first
        assert len(result) == 4
        assert set(result) == {'A', 'B', 'C', 'D'}
    
    def test_dfs_directed_simple(self):
        """Test DFS traversal on simple directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('A', 'D')
        
        result = g.dfs('A')
        assert result[0] == 'A'
        assert len(result) == 4
        assert set(result) == {'A', 'B', 'C', 'D'}
    
    def test_dfs_with_end_vertex(self):
        """Test DFS stops when end vertex is reached"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'D')
        g.add_edge('D', 'E')
        
        result = g.dfs('A', end='C')
        assert 'A' in result
        assert 'C' in result
        assert len(result) <= 4  # Should stop early
    
    def test_dfs_disconnected_components(self):
        """Test DFS only visits connected component"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('C', 'D')  # Separate component
        
        result = g.dfs('A')
        assert set(result) == {'A', 'B'}
        assert 'C' not in result
        assert 'D' not in result
    
    def test_dfs_single_vertex(self):
        """Test DFS with single isolated vertex"""
        g = Graph(directed=False, weighted=False)
        g.add_vertex('A')
        
        result = g.dfs('A')
        assert result == ['A']
    
    def test_dfs_nonexistent_vertex(self):
        """Test DFS with non-existent start vertex"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        
        result = g.dfs('Z')
        assert result == []
    
    def test_dfs_cyclic_graph(self):
        """Test DFS handles cycles correctly"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')  # Creates cycle
        
        result = g.dfs('A')
        assert len(result) == 3
        assert set(result) == {'A', 'B', 'C'}


class TestBFS:
    """Test cases for breadth-first search (BFS) traversal"""
    
    def test_bfs_undirected_simple(self):
        """Test BFS traversal on simple undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        
        result = g.bfs('A')
        assert result[0] == 'A'
        assert len(result) == 4
        assert set(result) == {'A', 'B', 'C', 'D'}
        # BFS should visit level-by-level: A first, then B,C, then D
        assert result.index('B') < result.index('D')
        assert result.index('C') < result.index('D')
    
    def test_bfs_directed_simple(self):
        """Test BFS traversal on simple directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        
        result = g.bfs('A')
        assert result[0] == 'A'
        assert len(result) == 4
    
    def test_bfs_with_end_vertex(self):
        """Test BFS stops when end vertex is reached"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        g.add_edge('C', 'E')
        
        result = g.bfs('A', end='D')
        assert 'A' in result
        assert 'D' in result
        assert len(result) <= 4  # Should stop when D is found
    
    def test_bfs_disconnected_components(self):
        """Test BFS only visits connected component"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('C', 'D')  # Separate component
        
        result = g.bfs('A')
        assert set(result) == {'A', 'B'}
        assert 'C' not in result
    
    def test_bfs_single_vertex(self):
        """Test BFS with single isolated vertex"""
        g = Graph(directed=False, weighted=False)
        g.add_vertex('A')
        
        result = g.bfs('A')
        assert result == ['A']
    
    def test_bfs_nonexistent_vertex(self):
        """Test BFS with non-existent start vertex"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        
        result = g.bfs('Z')
        assert result == []
    
    def test_bfs_level_order(self):
        """Test BFS traverses in correct level order"""
        g = Graph(directed=True, weighted=False)
        # Create tree structure: A -> B,C; B -> D,E; C -> F,G
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        g.add_edge('B', 'E')
        g.add_edge('C', 'F')
        g.add_edge('C', 'G')
        
        result = g.bfs('A')
        assert result[0] == 'A'  # Level 0
        # Level 1: B, C should come before level 2
        assert result.index('B') < result.index('D')
        assert result.index('C') < result.index('F')


class TestDijkstra:
    """Test cases for Dijkstra's shortest path algorithm"""
    
    def test_dijkstra_simple_weighted_graph(self):
        """Test Dijkstra on simple weighted graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 4)
        g.add_edge('A', 'C', 2)
        g.add_edge('C', 'B', 1)
        
        result = g.dijkstra('A')
        assert result['A'] == (0, ['A'])
        assert result['C'] == (2, ['A', 'C'])
        assert result['B'] == (3, ['A', 'C', 'B'])
    
    def test_dijkstra_directed_graph(self):
        """Test Dijkstra on directed weighted graph"""
        g = Graph(directed=True, weighted=True)
        g.add_edge('A', 'B', 5)
        g.add_edge('A', 'C', 2)
        g.add_edge('C', 'B', 1)
        # No edge from B to other vertices
        
        result = g.dijkstra('A')
        assert result['A'] == (0, ['A'])
        assert result['C'] == (2, ['A', 'C'])
        assert result['B'] == (3, ['A', 'C', 'B'])
    
    def test_dijkstra_unreachable_vertex(self):
        """Test Dijkstra with unreachable vertices"""
        g = Graph(directed=True, weighted=True)
        g.add_edge('A', 'B', 1)
        g.add_vertex('C')  # Isolated vertex
        
        result = g.dijkstra('A')
        assert result['A'] == (0, ['A'])
        assert result['B'] == (1, ['A', 'B'])
        assert result['C'] == (float('inf'), [])
    
    def test_dijkstra_complex_graph(self):
        """Test Dijkstra on complex weighted graph"""
        g = Graph(directed=False, weighted=True)
        edges = [
            ('A', 'B', 7), ('A', 'C', 9), ('A', 'F', 14),
            ('B', 'C', 10), ('B', 'D', 15), ('C', 'D', 11),
            ('C', 'F', 2), ('D', 'E', 6), ('E', 'F', 9)
        ]
        for u, v, w in edges:
            g.add_edge(u, v, w)
        
        result = g.dijkstra('A')
        assert result['A'] == (0, ['A'])
        assert result['C'] == (9, ['A', 'C'])
        assert result['F'] == (11, ['A', 'C', 'F'])
        assert result['E'] == (20, ['A', 'C', 'F', 'E'])
    
    def test_dijkstra_unweighted_graph_returns_empty(self):
        """Test Dijkstra returns empty dict for unweighted graphs"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        
        result = g.dijkstra('A')
        assert result == {}
    
    def test_dijkstra_nonexistent_start(self):
        """Test Dijkstra with non-existent start vertex"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 1)
        
        result = g.dijkstra('Z')
        assert result == {}
    
    def test_dijkstra_single_vertex(self):
        """Test Dijkstra with single vertex"""
        g = Graph(directed=False, weighted=True)
        g.add_vertex('A')
        
        result = g.dijkstra('A')
        assert result == {'A': (0, ['A'])}
    
    def test_dijkstra_all_vertices_reachable(self):
        """Test Dijkstra reaches all connected vertices"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 1)
        g.add_edge('B', 'C', 2)
        g.add_edge('C', 'D', 3)
        
        result = g.dijkstra('A')
        assert all(dist != float('inf') for dist, _ in result.values())
        assert len(result) == 4


class TestMinimumSpanningTree:
    """Test cases for MST algorithms (Kruskal and Prim)"""
    
    def test_mst_kruskal_simple_graph(self):
        """Test Kruskal's MST on simple weighted undirected graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 4)
        g.add_edge('A', 'C', 2)
        g.add_edge('B', 'C', 1)
        g.add_edge('B', 'D', 5)
        
        mst = g.minimum_spanning_tree_kruskal()
        assert mst is not None
        assert mst.vertex_count() == 4
        assert mst.edge_count() == 3  # MST has V-1 edges
        assert mst.total_weight() == 8  # 1 + 2 + 5
    
    def test_mst_prim_simple_graph(self):
        """Test Prim's MST on simple weighted undirected graph"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 4)
        g.add_edge('A', 'C', 2)
        g.add_edge('B', 'C', 1)
        g.add_edge('B', 'D', 5)
        
        mst = g.minimum_spanning_tree_prim()
        assert mst is not None
        assert mst.vertex_count() == 4
        assert mst.edge_count() == 3
        assert mst.total_weight() == 8  # Same as Kruskal
    
    def test_mst_both_algorithms_same_weight(self):
        """Test both MST algorithms produce same total weight"""
        g = Graph(directed=False, weighted=True)
        edges = [
            ('A', 'B', 7), ('A', 'C', 9), ('A', 'F', 14),
            ('B', 'C', 10), ('B', 'D', 15), ('C', 'D', 11),
            ('C', 'F', 2), ('D', 'E', 6), ('E', 'F', 9)
        ]
        for u, v, w in edges:
            g.add_edge(u, v, w)
        
        mst_kruskal = g.minimum_spanning_tree_kruskal()
        mst_prim = g.minimum_spanning_tree_prim()
        
        assert mst_kruskal.total_weight() == mst_prim.total_weight()
        assert mst_kruskal.edge_count() == 5  # 6 vertices - 1
        assert mst_prim.edge_count() == 5
    
    def test_mst_directed_graph_returns_none(self):
        """Test MST returns None for directed graphs"""
        g = Graph(directed=True, weighted=True)
        g.add_edge('A', 'B', 1)
        
        assert g.minimum_spanning_tree_kruskal() is None
        assert g.minimum_spanning_tree_prim() is None
    
    def test_mst_unweighted_graph_returns_none(self):
        """Test MST returns None for unweighted graphs"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        
        assert g.minimum_spanning_tree_kruskal() is None
        assert g.minimum_spanning_tree_prim() is None
    
    def test_mst_single_vertex(self):
        """Test MST with single vertex"""
        g = Graph(directed=False, weighted=True)
        g.add_vertex('A')
        
        mst = g.minimum_spanning_tree_kruskal()
        assert mst.vertex_count() == 1
        assert mst.edge_count() == 0
    
    def test_mst_empty_graph(self):
        """Test MST with empty graph"""
        g = Graph(directed=False, weighted=True)
        
        mst_k = g.minimum_spanning_tree_kruskal()
        mst_p = g.minimum_spanning_tree_prim()
        
        assert mst_k is not None
        assert mst_p is not None
        assert mst_k.vertex_count() == 0
        assert mst_p.vertex_count() == 0
    
    def test_mst_disconnected_graph(self):
        """Test MST with disconnected graph creates forest (Kruskal) or single component tree (Prim)"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 1)
        g.add_edge('C', 'D', 2)  # Separate component
        
        mst_k = g.minimum_spanning_tree_kruskal()
        mst_p = g.minimum_spanning_tree_prim()
        
        # Kruskal creates a spanning forest (all components)
        assert mst_k.edge_count() == 2
        assert mst_k.total_weight() == 3
        
        # Prim only creates MST for the component containing the start vertex
        # It will only span one connected component
        assert mst_p.edge_count() == 1
        assert mst_p.total_weight() in [1, 2]  # Either component A-B or C-D


class TestFindAllCycles:
    """Test cases for finding all cycles in graphs"""
    
    def test_find_cycles_undirected_simple(self):
        """Test finding cycles in simple undirected graph"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')  # Creates triangle
        
        cycles = g.find_all_cycles()
        assert len(cycles) > 0
        # Check that we found the triangle cycle
        found_triangle = False
        for cycle in cycles:
            if set(cycle[:-1]) == {'A', 'B', 'C'}:  # Ignore closing vertex
                found_triangle = True
        assert found_triangle
    
    def test_find_cycles_directed_simple(self):
        """Test finding cycles in simple directed graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')  # Creates directed cycle
        
        cycles = g.find_all_cycles()
        assert len(cycles) > 0
    
    def test_find_cycles_no_cycles(self):
        """Test finding cycles in acyclic graph"""
        g = Graph(directed=True, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'D')  # DAG - no cycles
        
        cycles = g.find_all_cycles()
        assert len(cycles) == 0
    
    def test_find_cycles_tree_no_cycles(self):
        """Test finding cycles in tree (undirected acyclic)"""
        g = Graph(directed=False, weighted=False)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')  # Tree structure
        
        cycles = g.find_all_cycles()
        assert len(cycles) == 0
    
    def test_find_cycles_multiple_cycles(self):
        """Test finding multiple cycles in graph"""
        g = Graph(directed=False, weighted=False)
        # Create two triangles sharing one edge
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')  # Triangle 1
        g.add_edge('C', 'D')
        g.add_edge('D', 'A')  # Triangle 2 with A-C shared
        
        cycles = g.find_all_cycles()
        assert len(cycles) >= 2  # At least 2 triangles
    
    def test_find_cycles_empty_graph(self):
        """Test finding cycles in empty graph"""
        g = Graph(directed=False, weighted=False)
        
        cycles = g.find_all_cycles()
        assert cycles == []
    
    def test_find_cycles_single_vertex(self):
        """Test finding cycles with single vertex"""
        g = Graph(directed=False, weighted=False)
        g.add_vertex('A')
        
        cycles = g.find_all_cycles()
        assert cycles == []
    
    def test_find_cycles_self_loop(self):
        """Test finding self-loop as cycle"""
        g = Graph(directed=True, weighted=False)
        g.add_vertex('A')
        g.add_edge('A', 'A')  # Self-loop
        
        cycles = g.find_all_cycles()
        # Self-loop should be detected as a cycle
        assert len(cycles) > 0


class TestIsTree:
    """Test cases for checking if a graph is a tree"""
    
    def test_empty_graph_is_tree(self):
        """Empty graph is considered a tree"""
        g = Graph(directed=False)
        assert g.is_tree() == True
    
    def test_single_vertex_is_tree(self):
        """Single vertex with no edges is a tree"""
        g = Graph(directed=False)
        g.add_vertex('A')
        assert g.is_tree() == True
    
    def test_simple_undirected_tree(self):
        """Simple undirected tree: A-B-C"""
        g = Graph(directed=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        
        assert g.is_tree() == True
        assert g.get_edge_count() == 2
        assert len(g.get_vertices()) == 3
    
    def test_undirected_tree_with_branching(self):
        """Undirected tree with branching"""
        g = Graph(directed=False)
        # Tree structure:
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 4)
        g.add_edge(2, 5)
        
        assert g.is_tree() == True
        assert g.get_edge_count() == 4
        assert len(g.get_vertices()) == 5
    
    def test_undirected_tree_becomes_cyclic(self):
        """Adding edge to tree creates cycle"""
        g = Graph(directed=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'D')
        
        assert g.is_tree() == True
        
        # Add edge to create cycle
        g.add_edge('D', 'A')
        assert g.is_tree() == False
    
    def test_undirected_disconnected_graph_not_tree(self):
        """Disconnected graph is not a tree"""
        g = Graph(directed=False)
        # Two separate components
        g.add_edge('A', 'B')
        g.add_edge('C', 'D')
        
        assert g.is_tree() == False
    
    def test_undirected_graph_with_cycle_not_tree(self):
        """Graph with cycle is not a tree"""
        g = Graph(directed=False)
        # Triangle
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        
        assert g.is_tree() == False
    
    def test_undirected_wrong_edge_count_not_tree(self):
        """Graph with wrong edge count is not a tree"""
        g = Graph(directed=False)
        # 3 nodes but 3 edges (should be 2 for tree)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        
        assert g.get_edge_count() == 3
        assert len(g.get_vertices()) == 3
        assert g.is_tree() == False  # m != n-1
    
    def test_directed_tree_simple(self):
        """Simple directed tree with root"""
        g = Graph(directed=True)
        # Root -> A -> B
        g.add_edge('Root', 'A')
        g.add_edge('A', 'B')
        
        assert g.is_tree() == True
    
    def test_directed_tree_with_branching(self):
        """Directed tree with branching from root"""
        g = Graph(directed=True)
        #       Root
        #       /  \
        #      A    B
        #     / \
        #    C   D
        g.add_edge('Root', 'A')
        g.add_edge('Root', 'B')
        g.add_edge('A', 'C')
        g.add_edge('A', 'D')
        
        assert g.is_tree() == True
        assert g.get_edge_count() == 4
        assert len(g.get_vertices()) == 5
    
    def test_directed_no_root_not_tree(self):
        """Directed graph with no root (no node with in-degree 0) is not a tree"""
        g = Graph(directed=True)
        # Cycle: A -> B -> C -> A
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        
        assert g.is_tree() == False
    
    def test_directed_multiple_roots_not_tree(self):
        """Directed graph with multiple roots is not a tree"""
        g = Graph(directed=True)
        # Two roots: A and C
        g.add_edge('A', 'B')
        g.add_edge('C', 'D')
        
        assert g.is_tree() == False
    
    def test_directed_with_cycle_not_tree(self):
        """Directed graph with cycle is not a tree"""
        g = Graph(directed=True)
        g.add_edge('Root', 'A')
        g.add_edge('A', 'B')
        g.add_edge('B', 'A')  # Cycle between A and B
        
        assert g.is_tree() == False
    
    def test_directed_node_with_multiple_parents_not_tree(self):
        """Directed graph where a node has multiple parents is not a tree"""
        g = Graph(directed=True)
        #   Root   A
        #     \   /
        #       B
        g.add_edge('Root', 'B')
        g.add_edge('A', 'B')  # B has two parents
        
        assert g.is_tree() == False  # B has in-degree 2
    
    def test_weighted_undirected_tree(self):
        """Weighted undirected graph can be a tree"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 1.5)
        g.add_edge('B', 'C', 2.0)
        g.add_edge('C', 'D', 3.5)
        
        assert g.is_tree() == True
    
    def test_weighted_directed_tree(self):
        """Weighted directed graph can be a tree"""
        g = Graph(directed=True, weighted=True)
        g.add_edge('Root', 'A', 10)
        g.add_edge('Root', 'B', 20)
        g.add_edge('A', 'C', 30)
        
        assert g.is_tree() == True
    
    def test_linear_chain_is_tree(self):
        """Linear chain is a tree"""
        g = Graph(directed=False)
        for i in range(10):
            g.add_edge(i, i+1)
        
        assert g.is_tree() == True
        assert g.get_edge_count() == 10
        assert len(g.get_vertices()) == 11
    
    def test_star_graph_is_tree(self):
        """Star graph (all nodes connected to center) is a tree"""
        g = Graph(directed=False)
        # Center connected to 5 outer nodes
        for i in range(1, 6):
            g.add_edge('Center', i)
        
        assert g.is_tree() == True
        assert g.get_edge_count() == 5
        assert len(g.get_vertices()) == 6
    
    def test_directed_star_is_tree(self):
        """Directed star graph from root is a tree"""
        g = Graph(directed=True)
        # Root pointing to 5 children
        for i in range(1, 6):
            g.add_edge('Root', i)
        
        assert g.is_tree() == True
    
    def test_get_edge_count_undirected(self):
        """Test edge counting for undirected graphs"""
        g = Graph(directed=False)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'D')
        
        assert g.get_edge_count() == 3
    
    def test_get_edge_count_directed(self):
        """Test edge counting for directed graphs"""
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'D')
        
        assert g.get_edge_count() == 3
    
    def test_get_edge_count_weighted(self):
        """Test edge counting for weighted graphs"""
        g = Graph(directed=False, weighted=True)
        g.add_edge('A', 'B', 1.0)
        g.add_edge('B', 'C', 2.0)
        
        assert g.get_edge_count() == 2
