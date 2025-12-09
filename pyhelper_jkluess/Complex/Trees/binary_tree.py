"""
Binary Tree Implementation

A binary tree is a tree where each node has at most two children,
referred to as the left child and right child.

Definitions:
- Binary Tree: Each node has max 2 children (left and right)
- Complete Binary Tree: Every node has either 0 or exactly 2 children
- Perfect Binary Tree: Complete tree where all leaves are at the same level (2^k - 1 nodes, k = h + 1)
- Balanced Binary Tree: Height difference between any two leaves is at most 1

Traversal Orders:
- Pre-Order: Root → Left → Right
- In-Order: Left → Root → Right  
- Post-Order: Left → Right → Root
"""

from typing import Any, List, Optional, Tuple
from collections import deque
try:
    from .tree import Tree, Node
except ImportError:
    from tree import Tree, Node

class BinaryNode(Node):
    """
    Node of a binary tree.
    
    Attributes:
        data: Data stored in the node
        parent: Pointer to parent node
        left: Pointer to left child node (stored as children[0])
        right: Pointer to right child node (stored as children[1])
    """
    
    def __init__(self, data: Any):
        """
        Creates a new binary tree node.
        
        Args:
            data: The data to store
        """
        super().__init__(data)
        # Binary nodes use children list but expose left/right properties
        # Store internal children list that may contain None
        self._children_internal: List[Optional['BinaryNode']] = []
    
    @property
    def children(self) -> List['BinaryNode']:
        """Get children list, filtering out None values for compatibility."""
        return [c for c in self._children_internal if c is not None]
    
    @children.setter
    def children(self, value: List['BinaryNode']) -> None:
        """Set children list."""
        self._children_internal = value
    
    @property
    def left(self) -> Optional['BinaryNode']:
        """Get the left child."""
        return self._children_internal[0] if len(self._children_internal) > 0 and self._children_internal[0] is not None else None
    
    @left.setter
    def left(self, node: Optional['BinaryNode']) -> None:
        """Set the left child."""
        if node is None:
            if len(self._children_internal) > 0:
                self._children_internal[0] = None
        else:
            if len(self._children_internal) == 0:
                self._children_internal.append(node)
            else:
                self._children_internal[0] = node
            node.parent = self
    
    @property
    def right(self) -> Optional['BinaryNode']:
        """Get the right child."""
        return self._children_internal[1] if len(self._children_internal) > 1 and self._children_internal[1] is not None else None
    
    @right.setter
    def right(self, node: Optional['BinaryNode']) -> None:
        """Set the right child."""
        if node is None:
            if len(self._children_internal) > 1:
                self._children_internal[1] = None
        else:
            # Ensure we have at least 2 slots in children
            while len(self._children_internal) < 2:
                self._children_internal.append(None)
            self._children_internal[1] = node
            node.parent = self
    
    def is_leaf(self) -> bool:
        """Checks if node is a leaf (has no children)."""
        return self.left is None and self.right is None
    
    def is_inner_node(self) -> bool:
        """Checks if node is an inner node (has at least one child)."""
        return self.left is not None or self.right is not None
    
    def has_left_child(self) -> bool:
        """Checks if node has a left child."""
        return self.left is not None
    
    def has_right_child(self) -> bool:
        """Checks if node has a right child."""
        return self.right is not None
    
    def has_both_children(self) -> bool:
        """Checks if node has both left and right children."""
        return self.left is not None and self.right is not None
    
    def children_count(self) -> int:
        """Returns the number of children (0, 1, or 2)."""
        count = 0
        if self.left is not None:
            count += 1
        if self.right is not None:
            count += 1
        return count
    
    def add_child(self, child: 'BinaryNode') -> None:
        """
        Adds a child node (validates binary constraint).
        
        Raises:
            ValueError: If node already has 2 children
        """
        if len(self.children) >= 2:
            raise ValueError("Binary node can have at most 2 children")
        if child not in self._children_internal:
            self._children_internal.append(child)
            child.parent = self
    
    def __str__(self) -> str:
        return f"BinaryNode({self.data})"
    
    def __repr__(self) -> str:
        left_data = self.left.data if self.left else None
        right_data = self.right.data if self.right else None
        return f"BinaryNode(data={self.data}, left={left_data}, right={right_data})"


class BinaryTree(Tree):
    """
    Binary Tree Implementation.
    
    A binary tree where each node has at most two children (left and right).
    Inherits all functionality from Tree class and adds binary-specific methods.
    """
    
    def __init__(self, root_data: Any = None):
        """
        Initialize a binary tree.
        
        Args:
            root_data: Optional data for the root node
        """
        super().__init__(root_data=None)  # Don't create root yet
        if root_data is not None:
            self.root = BinaryNode(root_data)
    
    def set_root(self, data: Any) -> BinaryNode:
        """
        Set or replace the root of the tree.
        
        Args:
            data: Data for the root node
            
        Returns:
            The root node
        """
        self.root = BinaryNode(data)
        return self.root
    
    def add_child(self, parent: BinaryNode, child_data: Any) -> BinaryNode:
        """
        Adds a child to a parent node (validates binary constraint).
        
        Args:
            parent: Parent node
            child_data: Data for the new child
            
        Returns:
            The created child node
            
        Raises:
            ValueError: If parent already has 2 children
        """
        if len([c for c in parent.children if c is not None]) >= 2:
            raise ValueError(f"Binary node can have at most 2 children. Use insert_left() or insert_right() instead.")
        
        child = BinaryNode(child_data)
        parent.add_child(child)
        return child
    
    def insert_left(self, parent: BinaryNode, data: Any) -> BinaryNode:
        """
        Insert a left child for the given parent node.
        
        Args:
            parent: The parent node
            data: Data for the new left child
            
        Returns:
            The newly created left child node
            
        Raises:
            ValueError: If parent already has a left child
        """
        if parent.left is not None:
            raise ValueError(f"Parent node already has a left child")
        
        child = BinaryNode(data)
        parent.left = child
        child.parent = parent
        return child
    
    def insert_right(self, parent: BinaryNode, data: Any) -> BinaryNode:
        """
        Insert a right child for the given parent node.
        
        Args:
            parent: The parent node
            data: Data for the new right child
            
        Returns:
            The newly created right child node
            
        Raises:
            ValueError: If parent already has a right child
        """
        if parent.right is not None:
            raise ValueError(f"Parent node already has a right child")
        
        child = BinaryNode(data)
        parent.right = child
        child.parent = parent
        return child
    
    # ==================== Tree Sorting Algorithm (BST) ====================
    
    def insert_sorted(self, value: Any) -> BinaryNode:
        """
        Insert a value using the tree sorting algorithm (Binary Search Tree insertion).
        
        Algorithm (Baum-Sortier-Algorithmus):
        1. If tree is empty, create root with the value
        2. Start at root and compare:
           - Go left if value < current node value
           - Go right if value > current node value
           - Ignore if value = current node value (no duplicates)
        3. Continue until finding an empty position
        4. Insert the value at that position
        
        Args:
            value: The value to insert (must be comparable)
            
        Returns:
            The newly created node
            
        Raises:
            TypeError: If values are not comparable
            
        Example:
            >>> tree = BinaryTree()
            >>> tree.insert_sorted(5)
            >>> tree.insert_sorted(3)
            >>> tree.insert_sorted(7)
            >>> tree.traverse_inorder()  # Returns sorted: [3, 5, 7]
        """
        if self.root is None:
            self.set_root(value)
            return self.root
        
        return self._insert_sorted_recursive(self.root, value)
    
    def _insert_sorted_recursive(self, node: BinaryNode, value: Any) -> Optional[BinaryNode]:
        """
        Helper method for recursive BST insertion.
        
        Args:
            node: Current node to compare with
            value: Value to insert
            
        Returns:
            The newly created node, or None if value is duplicate
        """
        try:
            if value < node.data:
                # Go left
                if node.left is None:
                    return self.insert_left(node, value)
                else:
                    return self._insert_sorted_recursive(node.left, value)
            elif value > node.data:
                # Go right
                if node.right is None:
                    return self.insert_right(node, value)
                else:
                    return self._insert_sorted_recursive(node.right, value)
            else:
                # value == node.data, ignore duplicates
                return None
        except TypeError as e:
            raise TypeError(f"Cannot compare values: {value} and {node.data}. Values must support <, >, = comparisons.") from e
    
    @classmethod
    def from_sorted_values(cls, values: List[Any]) -> 'BinaryTree':
        """
        Create a Binary Search Tree from a list of values using tree sorting algorithm.
        
        This method applies the Baum-Sortier-Algorithmus to sort values into a BST.
        Reading the tree In-Order will give the sorted sequence.
        
        Args:
            values: List of comparable values
            
        Returns:
            A new BinaryTree with values inserted using BST rules
            
        Example:
            >>> tree = BinaryTree.from_sorted_values([5, 3, 7, 1, 9, 4])
            >>> tree.traverse_inorder()  # Returns: [1, 3, 4, 5, 7, 9]
        """
        tree = cls()
        for value in values:
            tree.insert_sorted(value)
        return tree
    
    # ==================== Traversal Methods ====================
    
    def traverse_preorder(self, node: Optional[BinaryNode] = None) -> List[Any]:
        """
        Pre-order traversal: Root → Left → Right
        
        Args:
            node: Starting node (default: root)
            
        Returns:
            List of node data in pre-order
            
        Example:
            >>> tree = BinaryTree(1)
            >>> tree.insert_left(tree.root, 2)
            >>> tree.insert_right(tree.root, 3)
            >>> tree.traverse_preorder()
            [1, 2, 3]
        """
        if self.is_empty():
            return []
        
        if node is None:
            node = self.root
        
        result = []
        result.append(node.data)
        
        if node.left:
            result.extend(self.traverse_preorder(node.left))
        if node.right:
            result.extend(self.traverse_preorder(node.right))
        
        return result
    
    def traverse_inorder(self, node: Optional[BinaryNode] = None) -> List[Any]:
        """
        In-order traversal: Left → Root → Right
        
        Args:
            node: Starting node (default: root)
            
        Returns:
            List of node data in in-order
            
        Example:
            >>> tree = BinaryTree(1)
            >>> tree.insert_left(tree.root, 2)
            >>> tree.insert_right(tree.root, 3)
            >>> tree.traverse_inorder()
            [2, 1, 3]
        """
        if self.is_empty():
            return []
        
        if node is None:
            node = self.root
        
        result = []
        
        if node.left:
            result.extend(self.traverse_inorder(node.left))
        
        result.append(node.data)
        
        if node.right:
            result.extend(self.traverse_inorder(node.right))
        
        return result
    
    def traverse_postorder(self, node: Optional[BinaryNode] = None) -> List[Any]:
        """
        Post-order traversal: Left → Right → Root
        
        Args:
            node: Starting node (default: root)
            
        Returns:
            List of node data in post-order
            
        Example:
            >>> tree = BinaryTree(1)
            >>> tree.insert_left(tree.root, 2)
            >>> tree.insert_right(tree.root, 3)
            >>> tree.traverse_postorder()
            [2, 3, 1]
        """
        if self.is_empty():
            return []
        
        if node is None:
            node = self.root
        
        result = []
        
        if node.left:
            result.extend(self.traverse_postorder(node.left))
        if node.right:
            result.extend(self.traverse_postorder(node.right))
        
        result.append(node.data)
        
        return result
    
    def traverse_levelorder(self) -> List[Any]:
        """
        Level-order traversal (breadth-first).
        
        Returns:
            List of node data in level-order
        """
        if self.is_empty():
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            result.append(node.data)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return result
    
    # ==================== Tree Properties (inherited, some overridden for binary specifics) ====================
    
    def get_leaf_count(self) -> int:
        """
        Get the number of leaf nodes.
        
        Returns:
            Number of leaves
        """
        if self.is_empty():
            return 0
        return self._count_leaves(self.root)
    
    def _count_leaves(self, node: Optional[BinaryNode]) -> int:
        """Helper method to count leaf nodes recursively."""
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def get_height(self, node: Optional[BinaryNode] = None) -> int:
        """
        Get the height of the tree or subtree.
        Height is the maximum number of edges from node to a leaf.
        
        Args:
            node: Starting node (default: root)
            
        Returns:
            Height of the tree/subtree
        """
        if self.is_empty():
            return -1
        
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 0
        
        left_height = self.get_height(node.left) if node.left else -1
        right_height = self.get_height(node.right) if node.right else -1
        
        return 1 + max(left_height, right_height)
    
    # ==================== Special Binary Tree Checks ====================
    
    def is_complete(self) -> bool:
        """
        Check if the tree is a complete binary tree.
        A complete binary tree has every node with either 0 or exactly 2 children.
        
        Returns:
            True if complete, False otherwise
        """
        if self.is_empty():
            return True
        return self._is_complete_helper(self.root)
    
    def _is_complete_helper(self, node: Optional[BinaryNode]) -> bool:
        """Helper method to check if tree is complete."""
        if node is None:
            return True
        
        # A node must have either 0 or 2 children
        children_count = node.children_count()
        if children_count == 1:
            return False
        
        # Recursively check both subtrees
        return (self._is_complete_helper(node.left) and 
                self._is_complete_helper(node.right))
    
    def is_perfect(self) -> bool:
        """
        Check if the tree is a perfect binary tree.
        A perfect binary tree is complete and all leaves are at the same level.
        Has 2^k - 1 nodes where k = height + 1.
        
        Returns:
            True if perfect, False otherwise
        """
        if self.is_empty():
            return True
        
        if not self.is_complete():
            return False
        
        # Check if all leaves are at the same depth
        leaf_depths = self._get_all_leaf_depths(self.root, 0)
        return len(set(leaf_depths)) == 1
    
    def _get_all_leaf_depths(self, node: Optional[BinaryNode], depth: int) -> List[int]:
        """Helper method to get depths of all leaves."""
        if node is None:
            return []
        
        if node.is_leaf():
            return [depth]
        
        depths = []
        if node.left:
            depths.extend(self._get_all_leaf_depths(node.left, depth + 1))
        if node.right:
            depths.extend(self._get_all_leaf_depths(node.right, depth + 1))
        
        return depths
    
    def is_balanced(self) -> bool:
        """
        Check if the tree is balanced.
        A balanced tree has height difference between any two leaves at most 1.
        
        Returns:
            True if balanced, False otherwise
        """
        if self.is_empty():
            return True
        
        # Get all leaf depths
        leaf_depths = self._get_all_leaf_depths(self.root, 0)
        
        if not leaf_depths:
            return True
        
        min_depth = min(leaf_depths)
        max_depth = max(leaf_depths)
        
        return max_depth - min_depth <= 1
    
    # ==================== Utility Methods ====================
    
    def print_tree(self, node: Optional[BinaryNode] = None, prefix: str = "", is_left: bool = True) -> None:
        """
        Print the tree structure in a visual format.
        
        Args:
            node: Starting node (default: root)
            prefix: Prefix for the current line (used for recursion)
            is_left: Whether this node is a left child (used for recursion)
        """
        if self.is_empty():
            print("Empty tree")
            return
        
        if node is None:
            node = self.root
        
        if node is None:
            return
        
        print(prefix + ("├── " if is_left else "└── ") + str(node.data))
        
        # Print left and right children
        if node.left or node.right:
            if node.left:
                self.print_tree(node.left, prefix + ("│   " if is_left else "    "), True)
            else:
                print(prefix + ("│   " if is_left else "    ") + "├── None")
            
            if node.right:
                self.print_tree(node.right, prefix + ("│   " if is_left else "    "), False)
            else:
                print(prefix + ("│   " if is_left else "    ") + "└── None")
    
    def __str__(self) -> str:
        """String representation of the tree."""
        if self.is_empty():
            return "BinaryTree(empty)"
        node_count = self.get_node_count() if hasattr(super(), 'get_node_count') else self._count_nodes_binary(self.root)
        return f"BinaryTree(root={self.root.data}, nodes={node_count}, height={self.get_height()})"
    
    def _count_nodes_binary(self, node: Optional[BinaryNode]) -> int:
        """Helper to count nodes in binary tree."""
        if node is None:
            return 0
        return 1 + self._count_nodes_binary(node.left) + self._count_nodes_binary(node.right)
    
    def __repr__(self) -> str:
        """Detailed representation of the tree."""
        return self.__str__()
