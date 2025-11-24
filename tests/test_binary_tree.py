"""
Unit tests for BinaryTree class.
"""

import pytest
from pyhelper_jkluess.Complex.Trees.binary_tree import BinaryTree, BinaryNode


class TestBinaryNode:
    """Tests for BinaryNode class."""
    
    def test_node_creation(self):
        """Test creating a binary node."""
        node = BinaryNode(5)
        assert node.data == 5
        assert node.parent is None
        assert node.left is None
        assert node.right is None
    
    def test_is_leaf(self):
        """Test leaf node detection."""
        node = BinaryNode(1)
        assert node.is_leaf() is True
        
        node.left = BinaryNode(2)
        assert node.is_leaf() is False
    
    def test_is_inner_node(self):
        """Test inner node detection."""
        node = BinaryNode(1)
        assert node.is_inner_node() is False
        
        node.left = BinaryNode(2)
        assert node.is_inner_node() is True
    
    def test_has_left_child(self):
        """Test left child detection."""
        node = BinaryNode(1)
        assert node.has_left_child() is False
        
        node.left = BinaryNode(2)
        assert node.has_left_child() is True
    
    def test_has_right_child(self):
        """Test right child detection."""
        node = BinaryNode(1)
        assert node.has_right_child() is False
        
        node.right = BinaryNode(3)
        assert node.has_right_child() is True
    
    def test_has_both_children(self):
        """Test detection of both children."""
        node = BinaryNode(1)
        assert node.has_both_children() is False
        
        node.left = BinaryNode(2)
        assert node.has_both_children() is False
        
        node.right = BinaryNode(3)
        assert node.has_both_children() is True
    
    def test_children_count(self):
        """Test counting children."""
        node = BinaryNode(1)
        assert node.children_count() == 0
        
        node.left = BinaryNode(2)
        assert node.children_count() == 1
        
        node.right = BinaryNode(3)
        assert node.children_count() == 2


class TestBinaryTreeCreation:
    """Tests for binary tree creation."""
    
    def test_empty_tree(self):
        """Test creating an empty tree."""
        tree = BinaryTree()
        assert tree.is_empty() is True
        assert tree.root is None
    
    def test_tree_with_root(self):
        """Test creating tree with root."""
        tree = BinaryTree(1)
        assert tree.is_empty() is False
        assert tree.root is not None
        assert tree.root.data == 1
    
    def test_set_root(self):
        """Test setting root."""
        tree = BinaryTree()
        root = tree.set_root(10)
        assert tree.root.data == 10
        assert root.data == 10


class TestBinaryTreeInsertion:
    """Tests for inserting nodes."""
    
    def test_insert_left(self):
        """Test inserting left child."""
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        
        assert tree.root.left is not None
        assert tree.root.left.data == 2
        assert left.data == 2
        assert left.parent is tree.root
    
    def test_insert_right(self):
        """Test inserting right child."""
        tree = BinaryTree(1)
        right = tree.insert_right(tree.root, 3)
        
        assert tree.root.right is not None
        assert tree.root.right.data == 3
        assert right.data == 3
        assert right.parent is tree.root
    
    def test_insert_both_children(self):
        """Test inserting both children."""
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        right = tree.insert_right(tree.root, 3)
        
        assert tree.root.left.data == 2
        assert tree.root.right.data == 3
        assert tree.root.has_both_children()
    
    def test_insert_left_when_exists(self):
        """Test error when left child already exists."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        
        with pytest.raises(ValueError, match="already has a left child"):
            tree.insert_left(tree.root, 4)
    
    def test_insert_right_when_exists(self):
        """Test error when right child already exists."""
        tree = BinaryTree(1)
        tree.insert_right(tree.root, 3)
        
        with pytest.raises(ValueError, match="already has a right child"):
            tree.insert_right(tree.root, 5)


class TestTraversalPreorder:
    """Tests for pre-order traversal."""
    
    def test_empty_tree(self):
        """Test pre-order on empty tree."""
        tree = BinaryTree()
        assert tree.traverse_preorder() == []
    
    def test_single_node(self):
        """Test pre-order with single node."""
        tree = BinaryTree(1)
        assert tree.traverse_preorder() == [1]
    
    def test_left_child_only(self):
        """Test pre-order with only left child."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        assert tree.traverse_preorder() == [1, 2]
    
    def test_right_child_only(self):
        """Test pre-order with only right child."""
        tree = BinaryTree(1)
        tree.insert_right(tree.root, 3)
        assert tree.traverse_preorder() == [1, 3]
    
    def test_both_children(self):
        """Test pre-order with both children."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        assert tree.traverse_preorder() == [1, 2, 3]
    
    def test_complex_tree(self):
        """Test pre-order on more complex tree."""
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        
        assert tree.traverse_preorder() == [1, 2, 4, 5, 3]


class TestTraversalInorder:
    """Tests for in-order traversal."""
    
    def test_empty_tree(self):
        """Test in-order on empty tree."""
        tree = BinaryTree()
        assert tree.traverse_inorder() == []
    
    def test_single_node(self):
        """Test in-order with single node."""
        tree = BinaryTree(1)
        assert tree.traverse_inorder() == [1]
    
    def test_left_child_only(self):
        """Test in-order with only left child."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        assert tree.traverse_inorder() == [2, 1]
    
    def test_right_child_only(self):
        """Test in-order with only right child."""
        tree = BinaryTree(1)
        tree.insert_right(tree.root, 3)
        assert tree.traverse_inorder() == [1, 3]
    
    def test_both_children(self):
        """Test in-order with both children."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        assert tree.traverse_inorder() == [2, 1, 3]
    
    def test_complex_tree(self):
        """Test in-order on more complex tree."""
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        
        assert tree.traverse_inorder() == [4, 2, 5, 1, 3]


class TestTraversalPostorder:
    """Tests for post-order traversal."""
    
    def test_empty_tree(self):
        """Test post-order on empty tree."""
        tree = BinaryTree()
        assert tree.traverse_postorder() == []
    
    def test_single_node(self):
        """Test post-order with single node."""
        tree = BinaryTree(1)
        assert tree.traverse_postorder() == [1]
    
    def test_left_child_only(self):
        """Test post-order with only left child."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        assert tree.traverse_postorder() == [2, 1]
    
    def test_right_child_only(self):
        """Test post-order with only right child."""
        tree = BinaryTree(1)
        tree.insert_right(tree.root, 3)
        assert tree.traverse_postorder() == [3, 1]
    
    def test_both_children(self):
        """Test post-order with both children."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        assert tree.traverse_postorder() == [2, 3, 1]
    
    def test_complex_tree(self):
        """Test post-order on more complex tree."""
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        
        assert tree.traverse_postorder() == [4, 5, 2, 3, 1]


class TestTraversalLevelorder:
    """Tests for level-order traversal."""
    
    def test_empty_tree(self):
        """Test level-order on empty tree."""
        tree = BinaryTree()
        assert tree.traverse_levelorder() == []
    
    def test_single_node(self):
        """Test level-order with single node."""
        tree = BinaryTree(1)
        assert tree.traverse_levelorder() == [1]
    
    def test_complex_tree(self):
        """Test level-order on more complex tree."""
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        
        assert tree.traverse_levelorder() == [1, 2, 3, 4, 5]


class TestTreeProperties:
    """Tests for tree property methods."""
    
    def test_node_count_empty(self):
        """Test node count on empty tree."""
        tree = BinaryTree()
        assert tree.get_node_count() == 0
    
    def test_node_count_single(self):
        """Test node count with single node."""
        tree = BinaryTree(1)
        assert tree.get_node_count() == 1
    
    def test_node_count_multiple(self):
        """Test node count with multiple nodes."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        assert tree.get_node_count() == 3
    
    def test_height_empty(self):
        """Test height of empty tree."""
        tree = BinaryTree()
        assert tree.get_height() == -1
    
    def test_height_single(self):
        """Test height of single node."""
        tree = BinaryTree(1)
        assert tree.get_height() == 0
    
    def test_height_two_levels(self):
        """Test height with two levels."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        assert tree.get_height() == 1
    
    def test_height_three_levels(self):
        """Test height with three levels."""
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_left(left, 4)
        assert tree.get_height() == 2
    
    def test_leaf_count_empty(self):
        """Test leaf count on empty tree."""
        tree = BinaryTree()
        assert tree.get_leaf_count() == 0
    
    def test_leaf_count_single(self):
        """Test leaf count with single node."""
        tree = BinaryTree(1)
        assert tree.get_leaf_count() == 1
    
    def test_leaf_count_multiple(self):
        """Test leaf count with multiple nodes."""
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        
        assert tree.get_leaf_count() == 3  # Nodes 3, 4, 5


class TestCompleteTree:
    """Tests for complete binary tree detection."""
    
    def test_empty_tree(self):
        """Empty tree is complete."""
        tree = BinaryTree()
        assert tree.is_complete() is True
    
    def test_single_node(self):
        """Single node is complete."""
        tree = BinaryTree(1)
        assert tree.is_complete() is True
    
    def test_both_children(self):
        """Node with both children is complete."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        assert tree.is_complete() is True
    
    def test_left_child_only(self):
        """Node with only left child is not complete."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        assert tree.is_complete() is False
    
    def test_right_child_only(self):
        """Node with only right child is not complete."""
        tree = BinaryTree(1)
        tree.insert_right(tree.root, 3)
        assert tree.is_complete() is False
    
    def test_complete_three_levels(self):
        """Complete tree with three levels."""
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        
        assert tree.is_complete() is True
    
    def test_incomplete_three_levels(self):
        """Incomplete tree with three levels."""
        #       1
        #      / \
        #     2   3
        #    /
        #   4
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        
        assert tree.is_complete() is False


class TestPerfectTree:
    """Tests for perfect binary tree detection."""
    
    def test_empty_tree(self):
        """Empty tree is perfect."""
        tree = BinaryTree()
        assert tree.is_perfect() is True
    
    def test_single_node(self):
        """Single node is perfect."""
        tree = BinaryTree(1)
        assert tree.is_perfect() is True
    
    def test_perfect_two_levels(self):
        """Perfect tree with two levels (3 nodes)."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        assert tree.is_perfect() is True
    
    def test_not_perfect_uneven_leaves(self):
        """Not perfect - leaves at different levels."""
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        
        assert tree.is_perfect() is False
    
    def test_perfect_three_levels(self):
        """Perfect tree with three levels (7 nodes)."""
        #         1
        #       /   \
        #      2     3
        #     / \   / \
        #    4   5 6   7
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        right = tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        tree.insert_left(right, 6)
        tree.insert_right(right, 7)
        
        assert tree.is_perfect() is True
    
    def test_not_perfect_incomplete(self):
        """Not perfect - not complete."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        assert tree.is_perfect() is False


class TestBalancedTree:
    """Tests for balanced binary tree detection."""
    
    def test_empty_tree(self):
        """Empty tree is balanced."""
        tree = BinaryTree()
        assert tree.is_balanced() is True
    
    def test_single_node(self):
        """Single node is balanced."""
        tree = BinaryTree(1)
        assert tree.is_balanced() is True
    
    def test_balanced_two_levels(self):
        """Balanced tree with two levels."""
        tree = BinaryTree(1)
        tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        assert tree.is_balanced() is True
    
    def test_balanced_uneven(self):
        """Balanced tree with leaves at depth 1 and 2."""
        #       1
        #      / \
        #     2   3
        #    /
        #   4
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        
        assert tree.is_balanced() is True
    
    def test_not_balanced(self):
        """Not balanced - leaves at depth 1 and 3 (difference > 1)."""
        #       1
        #      / \
        #     2   3
        #    /
        #   4
        #  /
        # 5
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        tree.insert_right(tree.root, 3)
        left2 = tree.insert_left(left, 4)
        tree.insert_left(left2, 5)
        
        # Leaves are at depth 1 (node 3) and depth 3 (node 5)
        # Difference is 2, which is > 1, so not balanced
        assert tree.is_balanced() is False
    
    def test_balanced_perfect(self):
        """Perfect tree is balanced."""
        #         1
        #       /   \
        #      2     3
        #     / \   / \
        #    4   5 6   7
        tree = BinaryTree(1)
        left = tree.insert_left(tree.root, 2)
        right = tree.insert_right(tree.root, 3)
        tree.insert_left(left, 4)
        tree.insert_right(left, 5)
        tree.insert_left(right, 6)
        tree.insert_right(right, 7)
        
        assert tree.is_balanced() is True
