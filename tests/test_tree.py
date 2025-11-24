import pytest
from pyhelper_jkluess.Complex.Trees.tree import Tree, Node as TreeNode


class TestTreeNodeCreation:
    def test_create_node_with_data(self):
        """Test creating a tree node with data"""
        node = TreeNode("A")
        assert node.data == "A"
        assert node.parent is None
        assert node.children == []
    
    def test_create_node_with_number(self):
        """Test creating a tree node with numeric data"""
        node = TreeNode(42)
        assert node.data == 42
        assert node.parent is None
    
    def test_node_string_representation(self):
        """Test string representation of node"""
        node = TreeNode("Test")
        assert str(node) == "Node(Test)"
        assert "Node" in repr(node)


class TestTreeNodeOperations:
    def test_add_single_child(self):
        """Test adding a single child to a node"""
        parent = TreeNode("Parent")
        child = TreeNode("Child")
        parent.add_child(child)
        
        assert child in parent.children
        assert child.parent == parent
        assert len(parent.children) == 1
    
    def test_add_multiple_children(self):
        """Test adding multiple children to a node"""
        parent = TreeNode("Parent")
        child1 = TreeNode("Child1")
        child2 = TreeNode("Child2")
        child3 = TreeNode("Child3")
        
        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)
        
        assert len(parent.children) == 3
        assert child1.parent == parent
        assert child2.parent == parent
        assert child3.parent == parent
    
    def test_add_duplicate_child(self):
        """Test that adding same child twice doesn't duplicate"""
        parent = TreeNode("Parent")
        child = TreeNode("Child")
        
        parent.add_child(child)
        parent.add_child(child)
        
        assert len(parent.children) == 1
    
    def test_remove_child_success(self):
        """Test removing a child successfully"""
        parent = TreeNode("Parent")
        child = TreeNode("Child")
        parent.add_child(child)
        
        result = parent.remove_child(child)
        
        assert result is True
        assert child not in parent.children
        assert child.parent is None
    
    def test_remove_child_not_found(self):
        """Test removing a child that doesn't exist"""
        parent = TreeNode("Parent")
        child = TreeNode("Child")
        
        result = parent.remove_child(child)
        
        assert result is False
    
    def test_is_leaf(self):
        """Test leaf node detection"""
        leaf = TreeNode("Leaf")
        assert leaf.is_leaf() is True
        
        parent = TreeNode("Parent")
        child = TreeNode("Child")
        parent.add_child(child)
        
        assert parent.is_leaf() is False
        assert child.is_leaf() is True
    
    def test_is_inner_node(self):
        """Test inner node detection"""
        leaf = TreeNode("Leaf")
        assert leaf.is_inner_node() is False
        
        parent = TreeNode("Parent")
        child = TreeNode("Child")
        parent.add_child(child)
        
        assert parent.is_inner_node() is True
        assert child.is_inner_node() is False
    
    def test_degree(self):
        """Test degree calculation"""
        node = TreeNode("Node")
        assert node.degree() == 0
        
        node.add_child(TreeNode("Child1"))
        assert node.degree() == 1
        
        node.add_child(TreeNode("Child2"))
        node.add_child(TreeNode("Child3"))
        assert node.degree() == 3


class TestTreeCreation:
    def test_create_empty_tree(self):
        """Test creating an empty tree"""
        tree = Tree()
        assert tree.is_empty() is True
        assert tree.root is None
    
    def test_create_tree_with_root(self):
        """Test creating a tree with root data"""
        tree = Tree("Root")
        assert tree.is_empty() is False
        assert tree.root is not None
        assert tree.root.data == "Root"
    
    def test_set_root(self):
        """Test setting root on empty tree"""
        tree = Tree()
        root = tree.set_root("Root")
        
        assert tree.root is not None
        assert tree.root == root
        assert tree.root.data == "Root"
    
    def test_set_root_replaces_existing(self):
        """Test that set_root replaces existing root"""
        tree = Tree("OldRoot")
        new_root = tree.set_root("NewRoot")
        
        assert tree.root == new_root
        assert tree.root.data == "NewRoot"


class TestTreeAddChild:
    def test_add_child_to_root(self):
        """Test adding a child to root"""
        tree = Tree("Root")
        child = tree.add_child(tree.root, "Child")
        
        assert child in tree.root.children
        assert child.parent == tree.root
        assert child.data == "Child"
    
    def test_add_multiple_children(self):
        """Test adding multiple children"""
        tree = Tree("Root")
        child1 = tree.add_child(tree.root, "Child1")
        child2 = tree.add_child(tree.root, "Child2")
        
        assert len(tree.root.children) == 2
        assert child1 in tree.root.children
        assert child2 in tree.root.children
    
    def test_add_grandchildren(self):
        """Test adding children to children (grandchildren)"""
        tree = Tree("Root")
        child = tree.add_child(tree.root, "Child")
        grandchild = tree.add_child(child, "Grandchild")
        
        assert grandchild in child.children
        assert grandchild.parent == child


class TestTreeNodeCount:
    def test_empty_tree_node_count(self):
        """Test node count of empty tree"""
        tree = Tree()
        assert tree.get_node_count() == 0
    
    def test_single_node_tree(self):
        """Test node count with only root"""
        tree = Tree("Root")
        assert tree.get_node_count() == 1
    
    def test_tree_with_children(self):
        """Test node count with children"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        
        assert tree.get_node_count() == 4
    
    def test_tree_with_grandchildren(self):
        """Test node count with multiple levels"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        tree.add_child(child_b, "B1")
        
        assert tree.get_node_count() == 6


class TestTreeEdgeCount:
    def test_empty_tree_edge_count(self):
        """Test edge count of empty tree"""
        tree = Tree()
        assert tree.get_edge_count() == 0
    
    def test_single_node_edge_count(self):
        """Test edge count with only root"""
        tree = Tree("Root")
        assert tree.get_edge_count() == 0
    
    def test_tree_with_children_edge_count(self):
        """Test edge count with children"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        
        assert tree.get_edge_count() == 3


class TestTreeProperty:
    def test_verify_tree_property_empty(self):
        """Test tree property m = n - 1 on empty tree"""
        tree = Tree()
        # Empty tree: n=0, m=0, so 0 != 0-1, property is False
        assert tree.verify_tree_property() is False
    
    def test_verify_tree_property_single_node(self):
        """Test tree property on single node"""
        tree = Tree("Root")
        assert tree.verify_tree_property() is True
    
    def test_verify_tree_property_simple_tree(self):
        """Test tree property on simple tree"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        
        assert tree.get_node_count() == 4
        assert tree.get_edge_count() == 3
        assert tree.verify_tree_property() is True
    
    def test_verify_tree_property_complex_tree(self):
        """Test tree property on complex tree"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        tree.add_child(child_b, "B1")
        
        assert tree.get_node_count() == 7
        assert tree.get_edge_count() == 6
        assert tree.verify_tree_property() is True


class TestTreeDepth:
    def test_depth_of_root(self):
        """Test that root has depth 0"""
        tree = Tree("Root")
        assert tree.get_depth(tree.root) == 0
    
    def test_depth_of_children(self):
        """Test depth of direct children"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        
        assert tree.get_depth(child_a) == 1
        assert tree.get_depth(child_b) == 1
    
    def test_depth_of_grandchildren(self):
        """Test depth of grandchildren"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        grandchild = tree.add_child(child_a, "A1")
        
        assert tree.get_depth(grandchild) == 2
    
    def test_depth_none_for_nonexistent(self):
        """Test depth of node not in tree (has no parent)"""
        tree = Tree("Root")
        other_node = TreeNode("Other")
        
        # Node not in tree has no parent, so depth is 0
        assert tree.get_depth(other_node) == 0


class TestTreeHeight:
    def test_height_empty_tree(self):
        """Test height of empty tree"""
        tree = Tree()
        assert tree.get_height() == -1
    
    def test_height_single_node(self):
        """Test height of tree with only root"""
        tree = Tree("Root")
        assert tree.get_height() == 0
    
    def test_height_with_children(self):
        """Test height with one level of children"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        
        assert tree.get_height() == 1
    
    def test_height_with_grandchildren(self):
        """Test height with multiple levels"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        
        assert tree.get_height() == 2
    
    def test_height_unbalanced_tree(self):
        """Test height of unbalanced tree"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        child_a1 = tree.add_child(child_a, "A1")
        tree.add_child(child_a1, "A1a")
        
        # Height should be longest path
        assert tree.get_height() == 3


class TestTreeDegree:
    def test_degree_of_leaf(self):
        """Test degree of leaf node"""
        tree = Tree("Root")
        leaf = tree.add_child(tree.root, "Leaf")
        
        assert tree.get_degree(leaf) == 0
    
    def test_degree_of_root_with_children(self):
        """Test degree of root with multiple children"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        
        assert tree.get_degree(tree.root) == 3
    
    def test_degree_of_inner_node(self):
        """Test degree of inner node"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        
        assert tree.get_degree(child_a) == 2
    
    def test_degree_of_single_node_tree(self):
        """Test degree of root in single node tree"""
        tree = Tree("Root")
        
        assert tree.get_degree(tree.root) == 0
    
    def test_degree_changes_when_adding_children(self):
        """Test that degree updates when children are added"""
        tree = Tree("Root")
        
        assert tree.get_degree(tree.root) == 0
        
        tree.add_child(tree.root, "A")
        assert tree.get_degree(tree.root) == 1
        
        tree.add_child(tree.root, "B")
        assert tree.get_degree(tree.root) == 2
        
        tree.add_child(tree.root, "C")
        assert tree.get_degree(tree.root) == 3


class TestTreeLevels:
    def test_get_level_root(self):
        """Test getting level 0 (root)"""
        tree = Tree("Root")
        level_0 = tree.get_level(0)
        
        assert len(level_0) == 1
        assert level_0[0].data == "Root"
    
    def test_get_level_children(self):
        """Test getting level 1"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        
        level_1 = tree.get_level(1)
        level_data = [node.data for node in level_1]
        
        assert len(level_1) == 3
        assert "A" in level_data
        assert "B" in level_data
        assert "C" in level_data
    
    def test_get_level_nonexistent(self):
        """Test getting nonexistent level"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        
        level_5 = tree.get_level(5)
        assert level_5 == []
    
    def test_get_all_levels(self):
        """Test getting all levels"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        
        all_levels = tree.get_all_levels()
        
        assert len(all_levels) == 3
        assert len(all_levels[0]) == 1  # Root
        assert len(all_levels[1]) == 2  # A, B
        assert len(all_levels[2]) == 2  # A1, A2


class TestTreeClassification:
    def test_get_leaves_single_node(self):
        """Test getting leaves from single node tree"""
        tree = Tree("Root")
        leaves = tree.get_leaves()
        
        assert len(leaves) == 1
        assert leaves[0].data == "Root"
    
    def test_get_leaves_simple_tree(self):
        """Test getting leaves from simple tree"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        
        leaves = tree.get_leaves()
        leaf_data = [node.data for node in leaves]
        
        assert len(leaves) == 3
        assert "A" in leaf_data
        assert "B" in leaf_data
        assert "C" in leaf_data
        assert "Root" not in leaf_data
    
    def test_get_leaves_complex_tree(self):
        """Test getting leaves from complex tree"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        
        leaves = tree.get_leaves()
        leaf_data = [node.data for node in leaves]
        
        assert len(leaves) == 3
        assert "A1" in leaf_data
        assert "A2" in leaf_data
        assert "B" in leaf_data
        assert "A" not in leaf_data
        assert "Root" not in leaf_data
    
    def test_get_inner_nodes_single_node(self):
        """Test getting inner nodes from single node tree"""
        tree = Tree("Root")
        inner = tree.get_inner_nodes()
        
        assert len(inner) == 0
    
    def test_get_inner_nodes_simple_tree(self):
        """Test getting inner nodes from simple tree"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        
        inner = tree.get_inner_nodes()
        
        assert len(inner) == 1
        assert inner[0].data == "Root"
    
    def test_get_inner_nodes_complex_tree(self):
        """Test getting inner nodes from complex tree"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        
        inner = tree.get_inner_nodes()
        inner_data = [node.data for node in inner]
        
        assert len(inner) == 2
        assert "Root" in inner_data
        assert "A" in inner_data


class TestTreeTraversals:
    def test_preorder_traversal(self):
        """Test preorder traversal"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        tree.add_child(child_b, "B1")
        
        result = tree.traverse_preorder()
        
        # Preorder: Root, then children left-to-right, recursively
        assert result == ["Root", "A", "A1", "A2", "B", "B1"]
    
    def test_postorder_traversal(self):
        """Test postorder traversal"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        tree.add_child(child_b, "B1")
        
        result = tree.traverse_postorder()
        
        # Postorder: Children first, then node
        assert result == ["A1", "A2", "A", "B1", "B", "Root"]
    
    def test_levelorder_traversal(self):
        """Test level-order (breadth-first) traversal"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        tree.add_child(child_b, "B1")
        
        result = tree.traverse_levelorder()
        
        # Level-order: Level by level
        assert result == ["Root", "A", "B", "A1", "A2", "B1"]
    
    def test_traversal_empty_tree(self):
        """Test traversal of empty tree"""
        tree = Tree()
        
        assert tree.traverse_preorder() == []
        assert tree.traverse_postorder() == []
        assert tree.traverse_levelorder() == []
    
    def test_traversal_single_node(self):
        """Test traversal of single node tree"""
        tree = Tree("Root")
        
        assert tree.traverse_preorder() == ["Root"]
        assert tree.traverse_postorder() == ["Root"]
        assert tree.traverse_levelorder() == ["Root"]


class TestTreeSearch:
    def test_find_node_exists(self):
        """Test finding a node that exists"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(child_a, "A1")
        
        found = tree.find_node("A1")
        
        assert found is not None
        assert found.data == "A1"
    
    def test_find_node_root(self):
        """Test finding the root"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        
        found = tree.find_node("Root")
        
        assert found is not None
        assert found == tree.root
    
    def test_find_node_not_exists(self):
        """Test finding a node that doesn't exist"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        
        found = tree.find_node("Z")
        
        assert found is None
    
    def test_find_node_empty_tree(self):
        """Test finding in empty tree"""
        tree = Tree()
        
        found = tree.find_node("A")
        
        assert found is None


class TestTreeAncestorsDescendants:
    def test_get_ancestors_root(self):
        """Test getting ancestors of root"""
        tree = Tree("Root")
        ancestors = tree.get_ancestors(tree.root)
        
        assert ancestors == []
    
    def test_get_ancestors_child(self):
        """Test getting ancestors of direct child"""
        tree = Tree("Root")
        child = tree.add_child(tree.root, "A")
        
        ancestors = tree.get_ancestors(child)
        ancestor_data = [node.data for node in ancestors]
        
        assert len(ancestors) == 1
        assert "Root" in ancestor_data
    
    def test_get_ancestors_grandchild(self):
        """Test getting ancestors of grandchild"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        grandchild = tree.add_child(child_a, "A1")
        
        ancestors = tree.get_ancestors(grandchild)
        ancestor_data = [node.data for node in ancestors]
        
        assert len(ancestors) == 2
        assert "Root" in ancestor_data
        assert "A" in ancestor_data
    
    def test_get_descendants_leaf(self):
        """Test getting descendants of leaf"""
        tree = Tree("Root")
        leaf = tree.add_child(tree.root, "A")
        
        descendants = tree.get_descendants(leaf)
        
        assert descendants == []
    
    def test_get_descendants_with_children(self):
        """Test getting descendants of node with children"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        
        descendants = tree.get_descendants(child_a)
        desc_data = [node.data for node in descendants]
        
        assert len(descendants) == 2
        assert "A1" in desc_data
        assert "A2" in desc_data
    
    def test_get_descendants_with_grandchildren(self):
        """Test getting descendants including grandchildren"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_a1 = tree.add_child(child_a, "A1")
        tree.add_child(child_a1, "A1a")
        
        descendants = tree.get_descendants(child_a)
        desc_data = [node.data for node in descendants]
        
        assert len(descendants) == 2
        assert "A1" in desc_data
        assert "A1a" in desc_data


class TestTreePathFinding:
    def test_find_path_to_self(self):
        """Test finding path from node to itself"""
        tree = Tree("Root")
        child = tree.add_child(tree.root, "A")
        
        path = tree.find_path(child, child)
        
        assert path == [child]
    
    def test_find_path_parent_to_child(self):
        """Test finding path from parent to child"""
        tree = Tree("Root")
        child = tree.add_child(tree.root, "A")
        
        path = tree.find_path(tree.root, child)
        path_data = [node.data for node in path]
        
        assert len(path) == 2
        assert path_data == ["Root", "A"]
    
    def test_find_path_siblings(self):
        """Test finding path between siblings"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        
        path = tree.find_path(child_a, child_b)
        path_data = [node.data for node in path]
        
        assert len(path) == 3
        assert path_data == ["A", "Root", "B"]
    
    def test_find_path_cousins(self):
        """Test finding path between cousins"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        child_a1 = tree.add_child(child_a, "A1")
        child_b1 = tree.add_child(child_b, "B1")
        
        path = tree.find_path(child_a1, child_b1)
        path_data = [node.data for node in path]
        
        assert len(path) == 5
        assert path_data == ["A1", "A", "Root", "B", "B1"]


class TestTreeGraphProperties:
    def test_is_connected_single_node(self):
        """Test connectivity of single node tree"""
        tree = Tree("Root")
        assert tree.is_connected() is True
    
    def test_is_connected_multiple_nodes(self):
        """Test connectivity of tree with multiple nodes"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        
        assert tree.is_connected() is True
    
    def test_has_cycle(self):
        """Test that proper tree has no cycles"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        
        assert tree.has_cycle() is False
    
    def test_is_acyclic(self):
        """Test that tree is acyclic"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        tree.add_child(child_a, "A1")
        
        assert tree.is_acyclic() is True


class TestTreeStatistics:
    def test_statistics_empty_tree(self):
        """Test statistics of empty tree"""
        tree = Tree()
        stats = tree.get_statistics()
        
        assert stats['node_count'] == 0
        assert stats['edge_count'] == 0
        assert stats['height'] == -1
        assert stats['leaf_count'] == 0
        assert stats['inner_node_count'] == 0
    
    def test_statistics_single_node(self):
        """Test statistics of single node tree"""
        tree = Tree("Root")
        stats = tree.get_statistics()
        
        assert stats['node_count'] == 1
        assert stats['edge_count'] == 0
        assert stats['height'] == 0
        assert stats['leaf_count'] == 1
        assert stats['inner_node_count'] == 0
        assert stats['satisfies_tree_property'] is True
    
    def test_statistics_complex_tree(self):
        """Test statistics of complex tree"""
        tree = Tree("Root")
        child_a = tree.add_child(tree.root, "A")
        child_b = tree.add_child(tree.root, "B")
        tree.add_child(tree.root, "C")
        tree.add_child(child_a, "A1")
        tree.add_child(child_a, "A2")
        tree.add_child(child_b, "B1")
        
        stats = tree.get_statistics()
        
        assert stats['node_count'] == 7
        assert stats['edge_count'] == 6
        assert stats['height'] == 2
        assert stats['leaf_count'] == 4  # A1, A2, B1, C
        assert stats['inner_node_count'] == 3  # Root, A, B
        assert stats['satisfies_tree_property'] is True
        assert stats['is_connected'] is True
        assert stats['is_acyclic'] is True


class TestTreePrintTree:
    def test_print_tree_empty(self, capsys):
        """Test printing empty tree"""
        tree = Tree()
        tree.print_tree()
        
        captured = capsys.readouterr()
        assert "Empty tree" in captured.out
    
    def test_print_tree_single_node(self, capsys):
        """Test printing single node tree"""
        tree = Tree("Root")
        tree.print_tree()
        
        captured = capsys.readouterr()
        assert "Root" in captured.out
    
    def test_print_tree_with_children(self, capsys):
        """Test printing tree with children"""
        tree = Tree("Root")
        tree.add_child(tree.root, "A")
        tree.add_child(tree.root, "B")
        
        tree.print_tree()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Root" in output
        assert "A" in output
        assert "B" in output


class TestTreeFromAdjacencyMatrix:
    def test_create_from_simple_matrix(self):
        """Test creating tree from simple adjacency matrix"""
        # Tree: 0 -> 1, 0 -> 2
        matrix = [
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ]
        tree = Tree.from_adjacency_matrix(matrix)
        
        assert tree.root.data == 0
        assert tree.get_node_count() == 3
        assert tree.get_edge_count() == 2
        assert len(tree.root.children) == 2
        assert tree.verify_tree_property()
    
    def test_create_from_matrix_with_labels(self):
        """Test creating tree from matrix with custom labels"""
        matrix = [
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ]
        labels = ['A', 'B', 'C']
        tree = Tree.from_adjacency_matrix(matrix, labels)
        
        assert tree.root.data == 'A'
        assert tree.get_node_count() == 3
        children_data = [child.data for child in tree.root.children]
        assert 'B' in children_data
        assert 'C' in children_data
    
    def test_create_from_complex_matrix(self):
        """Test creating tree from complex adjacency matrix"""
        # Tree: 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
        matrix = [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        tree = Tree.from_adjacency_matrix(matrix)
        
        assert tree.get_node_count() == 5
        assert tree.get_edge_count() == 4
        assert tree.get_height() == 2
        assert tree.verify_tree_property()
    
    def test_empty_matrix(self):
        """Test creating tree from empty matrix"""
        matrix = []
        tree = Tree.from_adjacency_matrix(matrix)
        assert tree.is_empty()
    
    def test_non_square_matrix_raises_error(self):
        """Test that non-square matrix raises ValueError"""
        matrix = [
            [0, 1],
            [0, 0, 0]
        ]
        with pytest.raises(ValueError, match="must be square"):
            Tree.from_adjacency_matrix(matrix)
    
    def test_no_root_raises_error(self):
        """Test that matrix with no root raises ValueError"""
        # Circular: 0 -> 1, 1 -> 0
        matrix = [
            [0, 1],
            [1, 0]
        ]
        with pytest.raises(ValueError, match="No root found"):
            Tree.from_adjacency_matrix(matrix)
    
    def test_multiple_roots_raises_error(self):
        """Test that matrix with multiple roots raises ValueError"""
        # Two disconnected nodes, both could be roots
        matrix = [
            [0, 0],
            [0, 0]
        ]
        with pytest.raises(ValueError, match="Multiple roots"):
            Tree.from_adjacency_matrix(matrix)
    
    def test_multiple_parents_raises_error(self):
        """Test that node with multiple parents raises ValueError"""
        # Both 0 and 1 are parents of 2
        matrix = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0]
        ]
        # This creates multiple roots (0 and 1 both have no parents)
        with pytest.raises(ValueError, match="Multiple roots"):
            Tree.from_adjacency_matrix(matrix)
    
    def test_disconnected_graph_raises_error(self):
        """Test that disconnected graph raises ValueError"""
        # 0 -> 1, but 2 is disconnected (has no parent)
        matrix = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        # This creates multiple roots (0 and 2 both have no parents)
        with pytest.raises(ValueError, match="Multiple roots"):
            Tree.from_adjacency_matrix(matrix)
    
    def test_wrong_label_count_raises_error(self):
        """Test that wrong number of labels raises ValueError"""
        matrix = [
            [0, 1],
            [0, 0]
        ]
        labels = ['A']  # Only 1 label for 2 nodes
        with pytest.raises(ValueError, match="node_labels length"):
            Tree.from_adjacency_matrix(matrix, labels)


class TestTreeFromAdjacencyList:
    def test_create_from_simple_list(self):
        """Test creating tree from simple adjacency list"""
        adj_list = {
            'A': ['B', 'C'],
            'B': [],
            'C': []
        }
        tree = Tree.from_adjacency_list(adj_list, 'A')
        
        assert tree.root.data == 'A'
        assert tree.get_node_count() == 3
        assert tree.get_edge_count() == 2
        assert len(tree.root.children) == 2
    
    def test_create_from_complex_list(self):
        """Test creating tree from complex adjacency list"""
        adj_list = {
            'Root': ['A', 'B'],
            'A': ['A1', 'A2'],
            'B': ['B1'],
            'A1': [],
            'A2': [],
            'B1': []
        }
        tree = Tree.from_adjacency_list(adj_list, 'Root')
        
        assert tree.root.data == 'Root'
        assert tree.get_node_count() == 6
        assert tree.get_edge_count() == 5
        assert tree.get_height() == 2
        assert tree.verify_tree_property()
    
    def test_create_with_numeric_nodes(self):
        """Test creating tree with numeric node values"""
        adj_list = {
            1: [2, 3],
            2: [4],
            3: [],
            4: []
        }
        tree = Tree.from_adjacency_list(adj_list, 1)
        
        assert tree.root.data == 1
        assert tree.get_node_count() == 4
    
    def test_missing_children_in_list(self):
        """Test adjacency list where leaf nodes are omitted"""
        adj_list = {
            'Root': ['A', 'B'],
            'A': ['A1']
        }
        tree = Tree.from_adjacency_list(adj_list, 'Root')
        
        assert tree.get_node_count() == 4
        # B and A1 should be leaves even though not explicitly in adj_list
    
    def test_root_not_in_list_raises_error(self):
        """Test that missing root raises ValueError"""
        adj_list = {
            'A': ['B'],
            'B': []
        }
        with pytest.raises(ValueError, match="Root.*not found"):
            Tree.from_adjacency_list(adj_list, 'C')
    
    def test_cycle_raises_error(self):
        """Test that cycle in adjacency list raises ValueError"""
        adj_list = {
            'A': ['B'],
            'B': ['A']  # Creates cycle
        }
        with pytest.raises(ValueError, match="Cycle detected"):
            Tree.from_adjacency_list(adj_list, 'A')
    
    def test_duplicate_node_raises_error(self):
        """Test that duplicate node (appearing as child multiple times) raises error"""
        adj_list = {
            'Root': ['A', 'B'],
            'A': ['C'],
            'B': ['C']  # C appears twice (under A and B)
        }
        with pytest.raises(ValueError, match="Cycle detected"):
            Tree.from_adjacency_list(adj_list, 'Root')


class TestTreeCreationComparison:
    def test_same_tree_different_methods(self):
        """Test that all creation methods produce equivalent trees"""
        # Create tree manually
        tree1 = Tree('Root')
        a = tree1.add_child(tree1.root, 'A')
        b = tree1.add_child(tree1.root, 'B')
        tree1.add_child(a, 'A1')
        tree1.add_child(a, 'A2')
        
        # Create from adjacency matrix
        matrix = [
            [0, 1, 1, 0, 0],  # Root -> A, B
            [0, 0, 0, 1, 1],  # A -> A1, A2
            [0, 0, 0, 0, 0],  # B
            [0, 0, 0, 0, 0],  # A1
            [0, 0, 0, 0, 0]   # A2
        ]
        labels = ['Root', 'A', 'B', 'A1', 'A2']
        tree2 = Tree.from_adjacency_matrix(matrix, labels)
        
        # Create from adjacency list
        adj_list = {
            'Root': ['A', 'B'],
            'A': ['A1', 'A2'],
            'B': [],
            'A1': [],
            'A2': []
        }
        tree3 = Tree.from_adjacency_list(adj_list, 'Root')
        
        # All trees should have same structure
        assert tree1.get_node_count() == tree2.get_node_count() == tree3.get_node_count()
        assert tree1.get_edge_count() == tree2.get_edge_count() == tree3.get_edge_count()
        assert tree1.get_height() == tree2.get_height() == tree3.get_height()
        
        # Traversals should produce same node labels
        assert tree1.traverse_levelorder() == tree2.traverse_levelorder() == tree3.traverse_levelorder()


class TestTreeFromNestedStructure:
    def test_create_from_simple_nested_structure(self):
        """Test creating tree from simple nested structure"""
        # Tree: A with children B and C
        structure = ('A', ['B', 'C'])
        tree = Tree.from_nested_structure(structure)
        
        assert tree.root.data == 'A'
        assert tree.get_node_count() == 3
        assert tree.get_edge_count() == 2
        assert len(tree.root.children) == 2
        assert tree.root.children[0].data == 'B'
        assert tree.root.children[1].data == 'C'
    
    def test_create_from_single_value(self):
        """Test creating tree from single value (leaf)"""
        structure = 'Root'
        tree = Tree.from_nested_structure(structure)
        
        assert tree.root.data == 'Root'
        assert tree.get_node_count() == 1
        assert tree.is_empty() == False
    
    def test_create_math_expression_tree(self):
        """Test creating math expression tree with duplicate operators"""
        # Expression: (3 + 4) * 5 + 2 * 3
        # Both * operators and both 3 values should be distinct nodes
        structure = ('+', [
            ('*', [
                ('+', [3, 4]),
                5
            ]),
            ('*', [2, 3])
        ])
        
        tree = Tree.from_nested_structure(structure)
        
        # Root should be +
        assert tree.root.data == '+'
        
        # Should have 9 nodes: +, *, +, 3, 4, 5, *, 2, 3
        assert tree.get_node_count() == 9
        
        # Root should have 2 children (both *)
        assert len(tree.root.children) == 2
        assert tree.root.children[0].data == '*'
        assert tree.root.children[1].data == '*'
        
        # First * should have 2 children (+ and 5)
        first_multiply = tree.root.children[0]
        assert len(first_multiply.children) == 2
        assert first_multiply.children[0].data == '+'
        assert first_multiply.children[1].data == 5
        
        # The nested + should have 2 children (3 and 4)
        nested_plus = first_multiply.children[0]
        assert len(nested_plus.children) == 2
        assert nested_plus.children[0].data == 3
        assert nested_plus.children[1].data == 4
        
        # Second * should have 2 children (2 and 3)
        second_multiply = tree.root.children[1]
        assert len(second_multiply.children) == 2
        assert second_multiply.children[0].data == 2
        assert second_multiply.children[1].data == 3
    
    def test_multilevel_nested_structure(self):
        """Test creating deep nested structure"""
        structure = ('Root', [
            ('A', [
                ('A1', ['A1a', 'A1b']),
                'A2'
            ]),
            ('B', ['B1'])
        ])
        
        tree = Tree.from_nested_structure(structure)
        
        assert tree.root.data == 'Root'
        # Root, A, A1, A1a, A1b, A2, B, B1 = 8 nodes
        assert tree.get_node_count() == 8
        assert tree.get_height() == 3
    
    def test_duplicate_values_are_distinct_nodes(self):
        """Test that duplicate values create distinct nodes"""
        # Tree with three nodes all having value 'X'
        structure = ('X', [
            ('X', ['X'])
        ])
        
        tree = Tree.from_nested_structure(structure)
        
        # Should have 3 nodes despite same value
        assert tree.get_node_count() == 3
        
        # All three should have value 'X'
        assert tree.root.data == 'X'
        assert tree.root.children[0].data == 'X'
        assert tree.root.children[0].children[0].data == 'X'
        
        # But they should be different node objects
        assert tree.root is not tree.root.children[0]
        assert tree.root is not tree.root.children[0].children[0]
        assert tree.root.children[0] is not tree.root.children[0].children[0]
    
    def test_numeric_values_in_nested_structure(self):
        """Test nested structure with numeric values"""
        structure = (1, [
            (2, [4, 5]),
            (3, [6])
        ])
        
        tree = Tree.from_nested_structure(structure)
        
        assert tree.root.data == 1
        assert tree.get_node_count() == 6
        assert tree.root.children[0].data == 2
        assert tree.root.children[1].data == 3
    
    def test_empty_children_list(self):
        """Test node with empty children list"""
        structure = ('Root', [])
        tree = Tree.from_nested_structure(structure)
        
        assert tree.root.data == 'Root'
        assert tree.get_node_count() == 1
        assert len(tree.root.children) == 0
    
    def test_mixed_leaf_and_parent_nodes(self):
        """Test structure with mix of leaf values and parent nodes"""
        structure = ('Root', [
            'LeafA',  # Simple value
            ('ParentB', ['ChildB1', 'ChildB2']),  # Parent with children
            'LeafC'   # Simple value
        ])
        
        tree = Tree.from_nested_structure(structure)
        
        assert tree.get_node_count() == 6
        assert tree.root.children[0].data == 'LeafA'
        assert tree.root.children[0].is_leaf()
        assert tree.root.children[1].data == 'ParentB'
        assert not tree.root.children[1].is_leaf()
        assert len(tree.root.children[1].children) == 2


class TestToNestedStructure:
    def test_to_nested_structure_simple_tree(self):
        """Test converting simple tree to nested structure"""
        tree = Tree('A')
        tree.add_child(tree.root, 'B')
        tree.add_child(tree.root, 'C')
        
        structure = tree.to_nested_structure()
        
        # Should be ('A', ['B', 'C'])
        assert structure == ('A', ['B', 'C'])
    
    def test_to_nested_structure_single_node(self):
        """Test converting single node tree"""
        tree = Tree('Root')
        structure = tree.to_nested_structure()
        
        # Leaf node should just be the value
        assert structure == 'Root'
    
    def test_to_nested_structure_empty_tree(self):
        """Test converting empty tree"""
        tree = Tree()
        structure = tree.to_nested_structure()
        
        assert structure is None
    
    def test_to_nested_structure_multilevel(self):
        """Test converting multi-level tree"""
        tree = Tree('Root')
        a = tree.add_child(tree.root, 'A')
        tree.add_child(tree.root, 'B')
        tree.add_child(a, 'A1')
        tree.add_child(a, 'A2')
        
        structure = tree.to_nested_structure()
        
        # Should be ('Root', [('A', ['A1', 'A2']), 'B'])
        assert structure == ('Root', [('A', ['A1', 'A2']), 'B'])
    
    def test_to_nested_structure_with_duplicates(self):
        """Test converting tree with duplicate values"""
        tree = Tree('+')
        left = tree.add_child(tree.root, '*')
        tree.add_child(left, 3)
        tree.add_child(left, 4)
        right = tree.add_child(tree.root, '*')
        tree.add_child(right, 5)
        
        structure = tree.to_nested_structure()
        
        # Should be ('+', [('*', [3, 4]), ('*', [5])])
        assert structure == ('+', [('*', [3, 4]), ('*', [5])])
    
    def test_nested_structure_round_trip(self):
        """Test creating tree, exporting, and recreating"""
        # Create original tree with duplicate values
        tree1 = Tree('+')
        left = tree1.add_child(tree1.root, '*')
        plus_node = tree1.add_child(left, '+')
        tree1.add_child(plus_node, 3)
        tree1.add_child(plus_node, 4)
        tree1.add_child(left, 5)
        right = tree1.add_child(tree1.root, '*')
        tree1.add_child(right, 2)
        tree1.add_child(right, 3)
        
        # Export to nested structure
        structure = tree1.to_nested_structure()
        
        # Recreate from structure
        tree2 = Tree.from_nested_structure(structure)
        
        # Trees should be identical
        assert tree1.get_node_count() == tree2.get_node_count()
        assert tree1.get_height() == tree2.get_height()
        assert tree1.traverse_levelorder() == tree2.traverse_levelorder()
        
        # Export again should give same structure
        structure2 = tree2.to_nested_structure()
        assert structure == structure2
    
    def test_to_nested_structure_deep_tree(self):
        """Test converting deep nested tree"""
        tree = Tree(1)
        level1 = tree.add_child(tree.root, 2)
        level2 = tree.add_child(level1, 3)
        tree.add_child(level2, 4)
        
        structure = tree.to_nested_structure()
        
        # Should be (1, [(2, [(3, [4])])])
        assert structure == (1, [(2, [(3, [4])])])
    
    def test_to_nested_structure_preserves_order(self):
        """Test that child order is preserved"""
        tree = Tree('Root')
        tree.add_child(tree.root, 'First')
        tree.add_child(tree.root, 'Second')
        tree.add_child(tree.root, 'Third')
        
        structure = tree.to_nested_structure()
        
        assert structure == ('Root', ['First', 'Second', 'Third'])
    
    def test_to_nested_structure_math_expression(self):
        """Test converting complex math expression tree"""
        # Build: (3 + 4) * 5 + 2 * 3
        tree = Tree('+')
        left_mult = tree.add_child(tree.root, '*')
        left_plus = tree.add_child(left_mult, '+')
        tree.add_child(left_plus, 3)
        tree.add_child(left_plus, 4)
        tree.add_child(left_mult, 5)
        right_mult = tree.add_child(tree.root, '*')
        tree.add_child(right_mult, 2)
        tree.add_child(right_mult, 3)
        
        structure = tree.to_nested_structure()
        
        expected = ('+', [
            ('*', [
                ('+', [3, 4]),
                5
            ]),
            ('*', [2, 3])
        ])
        
        assert structure == expected


class TestGetAdjacencyMatrix:
    def test_get_adjacency_matrix_simple_tree(self):
        """Test getting adjacency matrix from a simple tree"""
        tree = Tree('Root')
        tree.add_child(tree.root, 'A')
        tree.add_child(tree.root, 'B')
        
        matrix = tree.get_adjacency_matrix()
        
        # Should be 3x3 matrix
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)
        
        # Root (0) should have children A (1) and B (2)
        assert matrix[0][1] == 1  # Root -> A
        assert matrix[0][2] == 1  # Root -> B
        assert matrix[1][0] == 0  # A does not point to Root
        assert matrix[2][0] == 0  # B does not point to Root
    
    def test_get_adjacency_matrix_empty_tree(self):
        """Test getting adjacency matrix from empty tree"""
        tree = Tree()
        matrix = tree.get_adjacency_matrix()
        assert matrix == []
    
    def test_get_adjacency_matrix_single_node(self):
        """Test getting adjacency matrix from single node tree"""
        tree = Tree('Root')
        matrix = tree.get_adjacency_matrix()
        
        assert len(matrix) == 1
        assert matrix[0] == [0]
    
    def test_get_adjacency_matrix_multilevel_tree(self):
        """Test getting adjacency matrix from multi-level tree"""
        tree = Tree('Root')
        a = tree.add_child(tree.root, 'A')
        b = tree.add_child(tree.root, 'B')
        a1 = tree.add_child(a, 'A1')
        a2 = tree.add_child(a, 'A2')
        
        matrix = tree.get_adjacency_matrix()
        
        # 5 nodes total
        assert len(matrix) == 5
        
        # Check parent-child relationships exist
        # Root should have 2 children
        assert sum(matrix[0]) == 2
        # A should have 2 children
        # Find A's index (should be 1 in BFS order)
        assert sum(matrix[1]) == 2
        # B should have 0 children
        assert sum(matrix[2]) == 0
    
    def test_adjacency_matrix_round_trip(self):
        """Test creating tree from matrix and exporting it back"""
        original_matrix = [
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        labels = ['A', 'B', 'C', 'D']
        
        tree = Tree.from_adjacency_matrix(original_matrix, labels)
        exported_matrix = tree.get_adjacency_matrix()
        
        # Should match original
        assert original_matrix == exported_matrix


class TestGetNodeLabels:
    def test_get_node_labels_simple_tree(self):
        """Test getting node labels from a simple tree"""
        tree = Tree('Root')
        tree.add_child(tree.root, 'A')
        tree.add_child(tree.root, 'B')
        
        labels = tree.get_node_labels()
        
        # Should have 3 labels in BFS order
        assert labels == ['Root', 'A', 'B']
    
    def test_get_node_labels_empty_tree(self):
        """Test getting node labels from empty tree"""
        tree = Tree()
        labels = tree.get_node_labels()
        assert labels == []
    
    def test_get_node_labels_single_node(self):
        """Test getting node labels from single node tree"""
        tree = Tree('Root')
        labels = tree.get_node_labels()
        assert labels == ['Root']
    
    def test_get_node_labels_multilevel_tree(self):
        """Test getting node labels from multi-level tree"""
        tree = Tree('Root')
        a = tree.add_child(tree.root, 'A')
        b = tree.add_child(tree.root, 'B')
        tree.add_child(a, 'A1')
        tree.add_child(a, 'A2')
        tree.add_child(b, 'B1')
        
        labels = tree.get_node_labels()
        
        # Should be in BFS order: Root, then level 1 (A, B), then level 2 (A1, A2, B1)
        assert labels == ['Root', 'A', 'B', 'A1', 'A2', 'B1']
    
    def test_get_node_labels_matches_adjacency_matrix_order(self):
        """Test that node labels order matches adjacency matrix rows/columns"""
        tree = Tree('X')
        y = tree.add_child(tree.root, 'Y')
        z = tree.add_child(tree.root, 'Z')
        tree.add_child(y, 'Y1')
        
        labels = tree.get_node_labels()
        matrix = tree.get_adjacency_matrix()
        
        # Labels should be in same order as matrix
        assert len(labels) == len(matrix)
        assert labels == ['X', 'Y', 'Z', 'Y1']
        
        # Verify matrix structure matches labels
        # X (index 0) should be parent of Y (index 1) and Z (index 2)
        assert matrix[0][1] == 1  # X -> Y
        assert matrix[0][2] == 1  # X -> Z
        # Y (index 1) should be parent of Y1 (index 3)
        assert matrix[1][3] == 1  # Y -> Y1
    
    def test_round_trip_with_get_node_labels(self):
        """Test that tree can be reconstructed using get_node_labels and get_adjacency_matrix"""
        # Create original tree
        tree1 = Tree('Root')
        a = tree1.add_child(tree1.root, 'A')
        b = tree1.add_child(tree1.root, 'B')
        tree1.add_child(a, 'A1')
        tree1.add_child(a, 'A2')
        tree1.add_child(b, 'B1')
        
        # Export
        matrix = tree1.get_adjacency_matrix()
        labels = tree1.get_node_labels()
        
        # Import into new tree
        tree2 = Tree.from_adjacency_matrix(matrix, labels)
        
        # Verify trees are identical
        assert tree1.get_node_count() == tree2.get_node_count()
        assert tree1.traverse_levelorder() == tree2.traverse_levelorder()
        
        # Export again and verify matrix/labels match
        matrix2 = tree2.get_adjacency_matrix()
        labels2 = tree2.get_node_labels()
        assert matrix == matrix2
        assert labels == labels2


class TestGetAdjacencyList:
    def test_get_adjacency_list_simple_tree(self):
        """Test getting adjacency list from a simple tree"""
        tree = Tree('Root')
        tree.add_child(tree.root, 'A')
        tree.add_child(tree.root, 'B')
        
        adj_list = tree.get_adjacency_list()
        
        # Should have 3 entries
        assert len(adj_list) == 3
        
        # Check structure
        assert 'Root' in adj_list
        assert 'A' in adj_list
        assert 'B' in adj_list
        
        # Check children
        assert 'A' in adj_list['Root']
        assert 'B' in adj_list['Root']
        assert len(adj_list['Root']) == 2
        assert adj_list['A'] == []
        assert adj_list['B'] == []
    
    def test_get_adjacency_list_empty_tree(self):
        """Test getting adjacency list from empty tree"""
        tree = Tree()
        adj_list = tree.get_adjacency_list()
        assert adj_list == {}
    
    def test_get_adjacency_list_single_node(self):
        """Test getting adjacency list from single node tree"""
        tree = Tree('Root')
        adj_list = tree.get_adjacency_list()
        
        assert len(adj_list) == 1
        assert 'Root' in adj_list
        assert adj_list['Root'] == []
    
    def test_get_adjacency_list_multilevel_tree(self):
        """Test getting adjacency list from multi-level tree"""
        tree = Tree('Root')
        a = tree.add_child(tree.root, 'A')
        b = tree.add_child(tree.root, 'B')
        a1 = tree.add_child(a, 'A1')
        a2 = tree.add_child(a, 'A2')
        
        adj_list = tree.get_adjacency_list()
        
        # 5 nodes total
        assert len(adj_list) == 5
        
        # Check structure
        assert adj_list['Root'] == ['A', 'B']
        assert adj_list['A'] == ['A1', 'A2']
        assert adj_list['B'] == []
        assert adj_list['A1'] == []
        assert adj_list['A2'] == []
    
    def test_adjacency_list_preserves_child_order(self):
        """Test that adjacency list preserves child order"""
        tree = Tree('Root')
        tree.add_child(tree.root, 'First')
        tree.add_child(tree.root, 'Second')
        tree.add_child(tree.root, 'Third')
        
        adj_list = tree.get_adjacency_list()
        
        assert adj_list['Root'] == ['First', 'Second', 'Third']
    
    def test_adjacency_list_round_trip(self):
        """Test creating tree from adjacency list and exporting it back"""
        original_list = {
            'Root': ['A', 'B'],
            'A': ['A1', 'A2'],
            'B': [],
            'A1': [],
            'A2': []
        }
        
        tree = Tree.from_adjacency_list(original_list, 'Root')
        exported_list = tree.get_adjacency_list()
        
        # Should match original
        assert original_list == exported_list
    
    def test_get_adjacency_list_with_numeric_data(self):
        """Test adjacency list with numeric node data"""
        tree = Tree(0)
        tree.add_child(tree.root, 1)
        tree.add_child(tree.root, 2)
        
        adj_list = tree.get_adjacency_list()
        
        assert adj_list[0] == [1, 2]
        assert adj_list[1] == []
        assert adj_list[2] == []
