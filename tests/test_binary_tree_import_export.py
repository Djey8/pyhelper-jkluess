"""
Tests for BinaryTree import/export methods (inherited from Tree).
"""

import pytest
from pyhelper_jkluess.Complex.Trees.binary_tree import BinaryTree


class TestBinaryTreeAdjacencyMatrix:
    """Tests for adjacency matrix import/export."""
    
    def test_get_adjacency_matrix_simple(self):
        """Test getting adjacency matrix from binary tree."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        
        matrix = tree.get_adjacency_matrix()
        
        # Check dimensions
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)
        
        # Check structure: 1 -> 2, 1 -> 3
        assert matrix[0][1] == 1  # 1 -> 2
        assert matrix[0][2] == 1  # 1 -> 3
        assert matrix[1][0] == 0  # No reverse edge
        assert matrix[2][0] == 0  # No reverse edge
    
    def test_from_adjacency_matrix_simple(self):
        """Test creating binary tree from adjacency matrix."""
        # Tree structure: 0 -> 1, 0 -> 2
        matrix = [
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ]
        labels = [1, 2, 3]
        
        tree = BinaryTree.from_adjacency_matrix(matrix, labels)
        
        assert tree.root.data == 1
        assert tree.get_node_count() == 3
        assert tree.root.left.data == 2
        assert tree.root.right.data == 3
    
    def test_adjacency_matrix_round_trip(self):
        """Test round-trip conversion via adjacency matrix."""
        # Create binary tree
        tree1 = BinaryTree(1)
        left = tree1.insert_left(tree1.root, 2)
        tree1.insert_right(tree1.root, 3)
        tree1.insert_left(left, 4)
        tree1.insert_right(left, 5)
        
        # Export to matrix
        matrix = tree1.get_adjacency_matrix()
        labels = tree1.get_node_labels()
        
        # Import back
        tree2 = BinaryTree.from_adjacency_matrix(matrix, labels)
        
        # Verify structure
        assert tree1.get_node_count() == tree2.get_node_count()
        assert tree1.traverse_levelorder() == tree2.traverse_levelorder()


class TestBinaryTreeAdjacencyList:
    """Tests for adjacency list import/export."""
    
    def test_get_adjacency_list_simple(self):
        """Test getting adjacency list from binary tree."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        
        adj_list = tree.get_adjacency_list()
        
        assert 1 in adj_list
        assert set(adj_list[1]) == {2, 3}
        assert adj_list[2] == []
        assert adj_list[3] == []
    
    def test_from_adjacency_list_simple(self):
        """Test creating binary tree from adjacency list."""
        adj_list = {
            1: [2, 3],
            2: [],
            3: []
        }
        
        tree = BinaryTree.from_adjacency_list(adj_list, root=1)
        
        assert tree.root.data == 1
        assert tree.get_node_count() == 3
        # Note: adjacency list doesn't preserve left/right order
        children_data = {tree.root.left.data, tree.root.right.data}
        assert children_data == {2, 3}
    
    def test_adjacency_list_round_trip(self):
        """Test round-trip conversion via adjacency list."""
        tree1 = BinaryTree('A')
        tree1.insert_left(tree1.root, 'B')
        tree1.insert_right(tree1.root, 'C')
        
        adj_list = tree1.get_adjacency_list()
        tree2 = BinaryTree.from_adjacency_list(adj_list, root='A')
        
        assert tree1.get_node_count() == tree2.get_node_count()
        assert set(tree1.traverse_levelorder()) == set(tree2.traverse_levelorder())


class TestBinaryTreeNestedStructure:
    """Tests for nested structure import/export."""
    
    def test_to_nested_structure_simple(self):
        """Test exporting binary tree to nested structure."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        
        nested = tree.to_nested_structure()
        
        assert nested == (1, [2, 3])
    
    def test_to_nested_structure_deep(self):
        """Test exporting deep binary tree."""
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        
        nested = tree.to_nested_structure()
        
        assert nested == (1, [(2, [4, 5]), 3])
    
    def test_from_nested_structure_simple(self):
        """Test creating binary tree from nested structure."""
        nested = (1, [2, 3])
        
        tree = BinaryTree.from_nested_structure(nested)
        
        assert tree.root.data == 1
        assert tree.get_node_count() == 3
        # Note: nested structure doesn't enforce left/right order in base Tree implementation
        children_data = [c.data for c in tree.root.children if c is not None]
        assert set(children_data) == {2, 3}
    
    def test_from_nested_structure_expression_tree(self):
        """Test creating expression tree from nested structure."""
        # Expression: (2 + 3) * 4
        nested = ('*', [('+', [2, 3]), 4])
        
        tree = BinaryTree.from_nested_structure(nested)
        
        assert tree.root.data == '*'
        assert tree.get_node_count() == 5
    
    def test_nested_structure_round_trip(self):
        """Test round-trip conversion via nested structure."""
        tree1 = BinaryTree('*')
        plus = tree1.insert_left(tree1.root, '+')
        tree1.insert_right(tree1.root, 4)
        tree1.insert_left(plus, 2)
        tree1.insert_right(plus, 3)
        
        nested = tree1.to_nested_structure()
        tree2 = BinaryTree.from_nested_structure(nested)
        
        assert tree1.get_node_count() == tree2.get_node_count()
        assert tree1.traverse_preorder() == tree2.traverse_preorder()
    
    def test_nested_structure_with_duplicates(self):
        """Test nested structure with duplicate values."""
        # Multiple nodes with same value (e.g., operators)
        nested = ('+', [('+', [1, 2]), ('+', [3, 4])])
        
        tree = BinaryTree.from_nested_structure(nested)
        
        # All '+' nodes should be distinct
        assert tree.get_node_count() == 7
        preorder = tree.traverse_preorder()
        assert preorder.count('+') == 3


class TestBinaryTreeGetNodeLabels:
    """Tests for get_node_labels method."""
    
    def test_get_node_labels_simple(self):
        """Test getting node labels from binary tree."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        
        labels = tree.get_node_labels()
        
        # Should be in BFS order
        assert labels == [1, 2, 3]
    
    def test_get_node_labels_matches_levelorder(self):
        """Test that labels match level-order traversal."""
        tree = BinaryTree('A')
        left = tree.insert_left(tree.root, 'B')
        right = tree.insert_right(tree.root, 'C')
        tree.insert_left(left, 'D')
        tree.insert_right(left, 'E')
        tree.insert_left(right, 'F')
        
        labels = tree.get_node_labels()
        levelorder = tree.traverse_levelorder()
        
        assert labels == levelorder
    
    def test_round_trip_with_labels_and_matrix(self):
        """Test complete round-trip with matrix and labels."""
        tree1 = BinaryTree(10)
        left = tree1.insert_left(tree1.root, 20)
        right = tree1.insert_right(tree1.root, 30)
        tree1.insert_left(left, 40)
        tree1.insert_right(left, 50)
        tree1.insert_right(right, 60)
        
        # Export
        matrix = tree1.get_adjacency_matrix()
        labels = tree1.get_node_labels()
        
        # Import
        tree2 = BinaryTree.from_adjacency_matrix(matrix, labels)
        
        # Verify identical structure (level-order and pre-order match)
        # Note: in-order may differ as adjacency matrix doesn't preserve left/right order
        assert tree1.traverse_levelorder() == tree2.traverse_levelorder()
        assert tree1.traverse_preorder() == tree2.traverse_preorder()
        assert tree1.get_height() == tree2.get_height()
        assert tree1.get_node_count() == tree2.get_node_count()
        # Verify same nodes, though in-order may vary
        assert set(tree1.traverse_inorder()) == set(tree2.traverse_inorder())


class TestBinaryTreeMixedImportExport:
    """Tests for mixing different import/export methods."""
    
    def test_matrix_to_nested(self):
        """Test converting matrix to nested structure."""
        matrix = [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        labels = [1, 2, 3, 4, 5]
        
        tree = BinaryTree.from_adjacency_matrix(matrix, labels)
        nested = tree.to_nested_structure()
        
        # Verify structure
        assert nested == (1, [(2, [4, 5]), 3])
    
    def test_nested_to_adjacency_list(self):
        """Test converting nested structure to adjacency list."""
        nested = ('A', ['B', ('C', ['D', 'E'])])
        
        tree = BinaryTree.from_nested_structure(nested)
        adj_list = tree.get_adjacency_list()
        
        assert 'A' in adj_list
        assert 'B' in adj_list
        assert 'C' in adj_list
        assert 'D' in adj_list.get('C', [])
        assert 'E' in adj_list.get('C', [])
    
    def test_all_formats_equivalent(self):
        """Test that all formats represent the same tree."""
        # Create original tree
        tree1 = BinaryTree(1)
        left = tree1.insert_left(tree1.root, 2)
        tree1.insert_right(tree1.root, 3)
        tree1.insert_left(left, 4)
        
        # Export to all formats
        matrix = tree1.get_adjacency_matrix()
        labels = tree1.get_node_labels()
        adj_list = tree1.get_adjacency_list()
        nested = tree1.to_nested_structure()
        
        # Import from all formats
        tree_from_matrix = BinaryTree.from_adjacency_matrix(matrix, labels)
        tree_from_list = BinaryTree.from_adjacency_list(adj_list, root=1)
        tree_from_nested = BinaryTree.from_nested_structure(nested)
        
        # All should have same structure
        node_count = tree1.get_node_count()
        assert tree_from_matrix.get_node_count() == node_count
        assert tree_from_list.get_node_count() == node_count
        assert tree_from_nested.get_node_count() == node_count
        
        # All should have same nodes (though order may vary for adj_list)
        original_nodes = set(tree1.traverse_levelorder())
        assert set(tree_from_matrix.traverse_levelorder()) == original_nodes
        assert set(tree_from_list.traverse_levelorder()) == original_nodes
        assert set(tree_from_nested.traverse_levelorder()) == original_nodes


class TestBinaryTreeEmptyCases:
    """Tests for empty tree edge cases."""
    
    def test_empty_tree_adjacency_matrix(self):
        """Test adjacency matrix for empty tree."""
        tree = BinaryTree()
        matrix = tree.get_adjacency_matrix()
        assert matrix == []
    
    def test_empty_tree_adjacency_list(self):
        """Test adjacency list for empty tree."""
        tree = BinaryTree()
        adj_list = tree.get_adjacency_list()
        assert adj_list == {}
    
    def test_empty_tree_nested_structure(self):
        """Test nested structure for empty tree."""
        tree = BinaryTree()
        nested = tree.to_nested_structure()
        assert nested is None
    
    def test_empty_tree_node_labels(self):
        """Test node labels for empty tree."""
        tree = BinaryTree()
        labels = tree.get_node_labels()
        assert labels == []
