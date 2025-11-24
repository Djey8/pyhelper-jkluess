"""
Unit tests for Tree to BinaryTree LCRS conversion
"""

import pytest
from pyhelper_jkluess.Complex.Trees import Tree, BinaryTree


class TestLCRSConversion:
    """Test LCRS (Left-Child Right-Sibling) conversion"""
    
    def test_empty_tree(self):
        """Test converting empty tree"""
        tree = Tree()
        binary = tree.to_binary_tree()
        assert binary.is_empty()
    
    def test_single_node(self):
        """Test converting single node tree"""
        tree = Tree("X")
        binary = tree.to_binary_tree()
        assert binary.root.data == "X"
        assert binary.root.left is None
        assert binary.root.right is None
    
    def test_binary_tree_lcrs_applied(self):
        """Test that LCRS is applied even for binary trees"""
        tree = Tree(1)
        left = tree.add_child(tree.root, 2)
        right = tree.add_child(tree.root, 3)
        tree.add_child(left, 4)
        tree.add_child(left, 5)
        
        binary = tree.to_binary_tree()
        
        # LCRS structure: left = first child, right = sibling
        assert binary.root.data == 1
        assert binary.root.left.data == 2  # First child
        assert binary.root.left.right.data == 3  # 2's sibling is 3
        assert binary.root.left.left.data == 4  # 2's first child is 4
        assert binary.root.left.left.right.data == 5  # 4's sibling is 5
        
        # All nodes should be preserved
        assert set(tree.traverse_preorder()) == set(binary.traverse_preorder())
    
    def test_three_children_lcrs(self):
        """Test LCRS conversion for node with 3 children"""
        tree = Tree("A")
        tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        tree.add_child(tree.root, "D")
        
        binary = tree.to_binary_tree()
        
        # Root's left child should be first child (B)
        assert binary.root.left.data == "B"
        
        # B's right sibling should be C
        assert binary.root.left.right.data == "C"
        
        # C's right sibling should be D
        assert binary.root.left.right.right.data == "D"
        
        # Root has no right child (only first child goes left)
        assert binary.root.right is None
    
    def test_complex_tree_with_mixed_nodes(self):
        """Test tree with both binary and non-binary subtrees"""
        tree = Tree("Root")
        a = tree.add_child(tree.root, "A")
        b = tree.add_child(tree.root, "B")
        c = tree.add_child(tree.root, "C")
        
        # A has 2 children (binary)
        tree.add_child(a, "A1")
        tree.add_child(a, "A2")
        
        # B has 3 children (needs LCRS)
        tree.add_child(b, "B1")
        tree.add_child(b, "B2")
        tree.add_child(b, "B3")
        
        binary = tree.to_binary_tree()
        
        # Check conversion worked
        assert binary.root.data == "Root"
        assert binary.root.left.data == "A"
        
        # A should preserve binary structure
        assert binary.root.left.left.data == "A1"
        
        # Pre-order should preserve all nodes
        tree_nodes = tree.traverse_preorder()
        binary_nodes = binary.traverse_preorder()
        
        # All nodes should be present
        assert set(tree_nodes) == set(binary_nodes)
    
    def test_deep_tree(self):
        """Test LCRS on deeper tree"""
        tree = Tree(1)
        child_2 = tree.add_child(tree.root, 2)
        child_3 = tree.add_child(tree.root, 3)
        child_4 = tree.add_child(tree.root, 4)
        
        tree.add_child(child_2, 5)
        tree.add_child(child_2, 6)
        tree.add_child(child_3, 7)
        
        binary = tree.to_binary_tree()
        
        # All nodes should be preserved
        tree_nodes = set(tree.traverse_preorder())
        binary_nodes = set(binary.traverse_preorder())
        assert tree_nodes == binary_nodes
    
    def test_node_count_preserved(self):
        """Test that node count is preserved after conversion"""
        tree = Tree("A")
        for i in range(5):
            tree.add_child(tree.root, f"Child{i}")
        
        binary = tree.to_binary_tree()
        
        assert tree.get_node_count() == binary.get_node_count()
    
    def test_lcrs_with_single_child(self):
        """Test LCRS with nodes having single child"""
        tree = Tree(1)
        child = tree.add_child(tree.root, 2)
        tree.add_child(child, 3)
        
        binary = tree.to_binary_tree()
        
        # Single child should become left child
        assert binary.root.left.data == 2
        assert binary.root.right is None
        assert binary.root.left.left.data == 3


class TestLCRSWithPreserveBinary:
    """Test LCRS conversion with preserve_binary=True"""
    
    def test_preserve_binary_subtree(self):
        """Test that binary subtrees are preserved when preserve_binary=True"""
        tree = Tree(1)
        left = tree.add_child(tree.root, 2)
        right = tree.add_child(tree.root, 3)
        tree.add_child(left, 4)
        tree.add_child(left, 5)
        
        binary = tree.to_binary_tree(preserve_binary=True)
        
        # Root has 2 children, should be preserved
        assert binary.root.data == 1
        assert binary.root.left.data == 2
        assert binary.root.right.data == 3
        
        # Node 2 also has 2 children, should be preserved
        assert binary.root.left.left.data == 4
        assert binary.root.left.right.data == 5
        
        # All nodes preserved
        assert set(tree.traverse_preorder()) == set(binary.traverse_preorder())
    
    def test_preserve_mixed_tree(self):
        """Test tree with both binary and non-binary nodes with preserve_binary=True
        
        Important: When a parent uses LCRS (has >2 children), child nodes cannot
        preserve their binary structure because the right pointer is used for siblings.
        """
        tree = Tree("Root")
        a = tree.add_child(tree.root, "A")
        b = tree.add_child(tree.root, "B")
        c = tree.add_child(tree.root, "C")  # Root has 3 children - needs LCRS
        
        # A has 2 children - would like to preserve, but can't because A.right is used for sibling B
        tree.add_child(a, "A1")
        tree.add_child(a, "A2")
        
        # B has 3 children - needs LCRS
        tree.add_child(b, "B1")
        tree.add_child(b, "B2")
        tree.add_child(b, "B3")
        
        binary = tree.to_binary_tree(preserve_binary=True)
        
        # Root has 3 children, uses LCRS
        assert binary.root.left.data == "A"
        assert binary.root.left.right.data == "B"
        assert binary.root.left.right.right.data == "C"
        
        # A has 2 children, but since A.right = B (sibling), A must also use LCRS
        # So A.left = A1, and A1.right = A2 (sibling chain)
        assert binary.root.left.left.data == "A1"
        assert binary.root.left.left.right.data == "A2"
        
        # B has 3 children, uses LCRS
        b_node = binary.root.left.right
        assert b_node.left.data == "B1"
        assert b_node.left.right.data == "B2"
        assert b_node.left.right.right.data == "B3"
        
        # All nodes preserved
        assert set(tree.traverse_preorder()) == set(binary.traverse_preorder())
    
    def test_preserve_vs_no_preserve(self):
        """Test difference between preserve_binary=True and False"""
        tree = Tree(1)
        tree.add_child(tree.root, 2)
        tree.add_child(tree.root, 3)
        
        # Without preserve
        binary_lcrs = tree.to_binary_tree(preserve_binary=False)
        assert binary_lcrs.root.left.data == 2
        assert binary_lcrs.root.left.right.data == 3
        assert binary_lcrs.root.right is None
        
        # With preserve
        binary_preserved = tree.to_binary_tree(preserve_binary=True)
        assert binary_preserved.root.left.data == 2
        assert binary_preserved.root.right.data == 3
        
        # Both preserve all nodes
        assert tree.get_node_count() == binary_lcrs.get_node_count()
        assert tree.get_node_count() == binary_preserved.get_node_count()
    
    def test_preserve_single_child(self):
        """Test preserve_binary with single-child nodes"""
        tree = Tree(1)
        child = tree.add_child(tree.root, 2)
        tree.add_child(child, 3)
        
        binary = tree.to_binary_tree(preserve_binary=True)
        
        # Single child should become left child (same as LCRS)
        assert binary.root.left.data == 2
        assert binary.root.right is None
        assert binary.root.left.left.data == 3
    
    def test_preserve_complex_hierarchy(self):
        """Test preserve_binary with complex mixed hierarchy
        
        When root has >2 children, all descendants use LCRS (can't preserve).
        """
        tree = Tree("A")
        b = tree.add_child(tree.root, "B")
        c = tree.add_child(tree.root, "C")
        d = tree.add_child(tree.root, "D")
        e = tree.add_child(tree.root, "E")  # 4 children - LCRS needed
        
        # B has 2 children - would like to preserve, but B.right is used for sibling C
        tree.add_child(b, "B1")
        tree.add_child(b, "B2")
        
        # C has 1 child
        c1 = tree.add_child(c, "C1")
        
        # C1 has 2 children - would like to preserve, but C1.right is used by C's parent structure
        tree.add_child(c1, "C1a")
        tree.add_child(c1, "C1b")
        
        binary = tree.to_binary_tree(preserve_binary=True)
        
        # All nodes should be preserved (node count)
        tree_nodes = set(tree.traverse_preorder())
        binary_nodes = set(binary.traverse_preorder())
        assert tree_nodes == binary_nodes
        
        # Check node count
        assert tree.get_node_count() == binary.get_node_count()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
