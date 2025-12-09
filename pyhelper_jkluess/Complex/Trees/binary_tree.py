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
    
    def iter_preorder(self, node: Optional[BinaryNode] = None):
        """
        Generator for preorder traversal.
        Yields nodes one at a time (memory efficient).
        
        Order: Root -> Left -> Right
        
        Args:
            node: Start node (default: root)
            
        Yields:
            Node data in preorder
            
        Example:
            >>> for value in tree.iter_preorder():
            ...     print(value)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return
        
        yield node.data
        
        if node.left:
            yield from self.iter_preorder(node.left)
        
        if node.right:
            yield from self.iter_preorder(node.right)
    
    def iter_inorder(self, node: Optional[BinaryNode] = None):
        """
        Generator for inorder traversal.
        Yields nodes one at a time (memory efficient).
        
        Order: Left -> Root -> Right
        
        Args:
            node: Start node (default: root)
            
        Yields:
            Node data in inorder (sorted order for BST)
            
        Example:
            >>> for value in tree.iter_inorder():
            ...     print(value)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return
        
        if node.left:
            yield from self.iter_inorder(node.left)
        
        yield node.data
        
        if node.right:
            yield from self.iter_inorder(node.right)
    
    def iter_postorder(self, node: Optional[BinaryNode] = None):
        """
        Generator for postorder traversal.
        Yields nodes one at a time (memory efficient).
        
        Order: Left -> Right -> Root
        
        Args:
            node: Start node (default: root)
            
        Yields:
            Node data in postorder
            
        Example:
            >>> for value in tree.iter_postorder():
            ...     print(value)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return
        
        if node.left:
            yield from self.iter_postorder(node.left)
        
        if node.right:
            yield from self.iter_postorder(node.right)
        
        yield node.data
    
    def iter_levelorder(self):
        """
        Generator for level-order (breadth-first) traversal.
        Yields nodes one at a time (memory efficient).
        
        Order: Level by level from top to bottom
        
        Yields:
            Node data in level-order
            
        Example:
            >>> for value in tree.iter_levelorder():
            ...     print(value)
        """
        if self.is_empty():
            return
        
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            yield node.data
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
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


if __name__ == "__main__":
    print("=" * 80)
    print("EXAMPLE 1: Create Binary Tree Manually")
    print("=" * 80)
    
    # Create binary tree
    tree = BinaryTree(1)
    
    # Add children using binary-specific methods
    left_child = tree.insert_left(tree.root, 2)
    right_child = tree.insert_right(tree.root, 3)
    
    # Add children to left subtree
    tree.insert_left(left_child, 4)
    tree.insert_right(left_child, 5)
    
    # Add children to right subtree
    tree.insert_left(right_child, 6)
    tree.insert_right(right_child, 7)
    
    # Visualization
    print("=== Binary Tree Structure ===")
    tree.print_tree()
    
    print("\n=== Binary Tree Statistics ===")
    stats = tree.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n=== Binary Tree Properties ===")
    print(f"Is Complete: {tree.is_complete()}")
    print(f"Is Perfect: {tree.is_perfect()}")
    print(f"Is Balanced: {tree.is_balanced()}")
    print(f"Leaf Count: {tree.get_leaf_count()}")
    
    print("\n=== Binary Tree Traversals ===")
    print(f"Pre-order (Root→Left→Right): {tree.traverse_preorder()}")
    print(f"In-order (Left→Root→Right): {tree.traverse_inorder()}")
    print(f"Post-order (Left→Right→Root): {tree.traverse_postorder()}")
    print(f"Level-order (Breadth-first): {tree.traverse_levelorder()}")
    
    print("\n=== Inherited Tree Methods ===")
    print(f"Node Count: {tree.get_node_count()}")
    print(f"Edge Count: {tree.get_edge_count()}")
    print(f"Height: {tree.get_height()}")
    print(f"Tree Property (m = n - 1): {tree.verify_tree_property()}")
    
    print("=" * 80)
    print("EXAMPLE 2: Binary Tree from Nested Structure (RECOMMENDED)")
    print("=" * 80)
    
    # Create binary tree from nested structure
    # Format: (root, [left_subtree, right_subtree])
    # Leaf nodes can be just values, or wrapped in () for consistency
    structure = (1, [(2, [4, 5]), (3, [6, 7])])
    tree2 = BinaryTree.from_nested_structure(structure)
    
    print(f"\nNested Structure: {structure}")
    print("\nTree Structure:")
    tree2.print_tree()
    
    print("\nTraversals:")
    print(f"Pre-order: {tree2.traverse_preorder()}")
    print(f"In-order: {tree2.traverse_inorder()}")
    print(f"Post-order: {tree2.traverse_postorder()}")
    
    print("=" * 80)
    print("EXAMPLE 3: Expression Tree with Duplicate Values")
    print("=" * 80)
    
    # Mathematical expression: (3 + 5) * 3
    # Note: Two different nodes with value 3
    expr_structure = ('*', [('+', [3, 5]), 3])
    expr_tree = BinaryTree.from_nested_structure(expr_structure)
    
    print(f"\nExpression: (3 + 5) * 3")
    print(f"Nested Structure: {expr_structure}")
    print("\nExpression Tree:")
    expr_tree.print_tree()
    
    print("\nTraversals:")
    print(f"Pre-order (Prefix): {expr_tree.traverse_preorder()}")
    print(f"In-order (Infix): {expr_tree.traverse_inorder()}")
    print(f"Post-order (Postfix): {expr_tree.traverse_postorder()}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Create Binary Tree from Adjacency Matrix")
    print("=" * 80)
    
    # Define adjacency matrix for binary tree
    # Tree structure: 1 -> 2, 1 -> 3, 2 -> 4, 2 -> 5
    matrix = [
        [0, 1, 1, 0, 0],  # Node 1 has children 2, 3
        [0, 0, 0, 1, 1],  # Node 2 has children 4, 5
        [0, 0, 0, 0, 0],  # Node 3 has no children
        [0, 0, 0, 0, 0],  # Node 4 has no children
        [0, 0, 0, 0, 0]   # Node 5 has no children
    ]
    labels = [1, 2, 3, 4, 5]
    
    tree3 = BinaryTree.from_adjacency_matrix(matrix, labels)
    
    print("\nAdjacency Matrix:")
    for i, row in enumerate(matrix):
        print(f"  Node {labels[i]}: {row}")
    
    print("\nTree Structure:")
    tree3.print_tree()
    
    print("\n⚠️ Note: Adjacency matrix may not preserve left/right order!")
    print("For exact left/right placement, use nested structure format.")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Create Binary Tree from Adjacency List")
    print("=" * 80)
    
    # Define adjacency list
    adj_list = {
        1: [2, 3],
        2: [4, 5],
        3: [],
        4: [],
        5: []
    }
    
    tree4 = BinaryTree.from_adjacency_list(adj_list, root=1)
    
    print("\nAdjacency List:")
    for parent, children in adj_list.items():
        print(f"  {parent}: {children}")
    
    print("\nTree Structure:")
    tree4.print_tree()
    
    print("\nBinary Properties:")
    print(f"Is Complete: {tree4.is_complete()}")
    print(f"Is Perfect: {tree4.is_perfect()}")
    print(f"Is Balanced: {tree4.is_balanced()}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Export Binary Tree to Different Formats")
    print("=" * 80)
    
    # Use tree from Example 1
    print("\nOriginal Binary Tree:")
    tree.print_tree()
    
    # Export to nested structure (best for binary trees)
    nested = tree.to_nested_structure()
    print(f"\n1. Nested Structure (preserves left/right):")
    print(f"   {nested}")
    
    # Export to adjacency matrix
    matrix_export = tree.get_adjacency_matrix()
    print(f"\n2. Adjacency Matrix:")
    for i, row in enumerate(matrix_export):
        print(f"   Row {i}: {row}")
    
    # Export to adjacency list
    adj_list_export = tree.get_adjacency_list()
    print(f"\n3. Adjacency List:")
    for parent, children in adj_list_export.items():
        print(f"   {parent}: {children}")
    
    # Get node labels
    labels_export = tree.get_node_labels()
    print(f"\n4. Node Labels (level-order):")
    print(f"   {labels_export}")
    
    print("=" * 80)
    print("EXAMPLE 7: Round-trip Conversion (Nested Structure)")
    print("=" * 80)
    
    original_structure = (10, [(20, [40, 50]), (30, [60, 70])])
    print(f"Original Structure: {original_structure}")
    
    # Create tree
    tree_rt = BinaryTree.from_nested_structure(original_structure)
    print("\nTree from Structure:")
    tree_rt.print_tree()
    
    # Export back to structure
    exported_structure = tree_rt.to_nested_structure()
    print(f"\nExported Structure: {exported_structure}")
    
    # Verify they match
    structures_match = original_structure == exported_structure
    print(f"\nStructures match: {structures_match}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Binary Tree Validation - Non-Binary Structure")
    print("=" * 80)
    
    print("\nAttempting to create binary tree with 3 children:")
    try:
        # This should fail - node can't have 3 children
        invalid_structure = (1, [(2,), (3,), (4,)])
        invalid_tree = BinaryTree.from_nested_structure(invalid_structure)
        print("✗ Unexpectedly succeeded!")
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    print("\nAttempting to add 3rd child to a node:")
    try:
        test_tree = BinaryTree(1)
        test_tree.insert_left(test_tree.root, 2)
        test_tree.insert_right(test_tree.root, 3)
        # This should fail
        test_tree.add_child(test_tree.root, 4)
        print("✗ Unexpectedly succeeded!")
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Different Binary Tree Types")
    print("=" * 80)
    
    # Complete binary tree
    print("\n1. Complete Binary Tree:")
    complete = BinaryTree.from_nested_structure((1, [(2, [4, 5]), 3]))
    complete.print_tree()
    print(f"   Is Complete: {complete.is_complete()}")
    print(f"   Is Perfect: {complete.is_perfect()}")
    print(f"   Is Balanced: {complete.is_balanced()}")
    
    # Perfect binary tree
    print("\n2. Perfect Binary Tree:")
    perfect = BinaryTree.from_nested_structure((1, [(2, [4, 5]), (3, [6, 7])]))
    perfect.print_tree()
    print(f"   Is Complete: {perfect.is_complete()}")
    print(f"   Is Perfect: {perfect.is_perfect()}")
    print(f"   Is Balanced: {perfect.is_balanced()}")
    
    # Unbalanced binary tree (left-heavy chain)
    print("\n3. Unbalanced Binary Tree:")
    unbalanced = BinaryTree.from_nested_structure((1, [(2, [(3, [4])])]))
    unbalanced.print_tree()
    print(f"   Is Complete: {unbalanced.is_complete()}")
    print(f"   Is Perfect: {unbalanced.is_perfect()}")
    print(f"   Is Balanced: {unbalanced.is_balanced()}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Tree Sorting Algorithm (Baum-Sortier-Algorithmus)")
    print("=" * 80)
    
    print("\nSorting Algorithm - Inserting values one by one:")
    unsorted = [5, 3, 7, 1, 9, 4, 6]
    print(f"Values to insert: {unsorted}")
    
    sort_tree = BinaryTree()
    for val in unsorted:
        sort_tree.insert_sorted(val)
        print(f"  Inserted {val}")
    
    print("\nResulting Binary Search Tree:")
    sort_tree.print_tree()
    
    print("\nTraversals:")
    print(f"  Pre-order:  {sort_tree.traverse_preorder()}")
    print(f"  In-order:   {sort_tree.traverse_inorder()}  ← Sorted!")
    print(f"  Post-order: {sort_tree.traverse_postorder()}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 11: Batch Sorting with from_sorted_values()")
    print("=" * 80)
    
    unsorted_batch = [8, 3, 10, 1, 6, 14, 4, 7, 13]
    print(f"\nInput (unsorted): {unsorted_batch}")
    
    bst = BinaryTree.from_sorted_values(unsorted_batch)
    
    print("\nBinary Search Tree structure:")
    bst.print_tree()
    
    sorted_output = bst.traverse_inorder()
    print(f"\nSorted output (In-order): {sorted_output}")
    print(f"Python sorted():          {sorted(unsorted_batch)}")
    print(f"Match: {sorted_output == sorted(unsorted_batch)}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 12: Sorting with Duplicates")
    print("=" * 80)
    
    with_dupes = [5, 3, 7, 3, 5, 9, 1, 7]
    print(f"\nInput with duplicates: {with_dupes}")
    
    bst_dupes = BinaryTree.from_sorted_values(with_dupes)
    
    print("\nTree structure (duplicates ignored):")
    bst_dupes.print_tree()
    
    result = bst_dupes.traverse_inorder()
    unique_sorted = sorted(set(with_dupes))
    
    print(f"\nOutput: {result}")
    print(f"Unique sorted: {unique_sorted}")
    print(f"Match: {result == unique_sorted}")
    print("\nNote: Duplicates are ignored as per algorithm specification (k₁ = k₀)")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 13: Sorting Different Data Types")
    print("=" * 80)
    
    # Sorting strings
    print("\n1. Sorting strings (lexicographic order):")
    words = ["dog", "cat", "elephant", "ant", "bear"]
    print(f"   Input: {words}")
    
    word_tree = BinaryTree.from_sorted_values(words)
    word_result = word_tree.traverse_inorder()
    
    print(f"   Sorted: {word_result}")
    
    # Sorting floats
    print("\n2. Sorting floats:")
    floats = [3.14, 2.71, 1.41, 2.0, 3.0]
    print(f"   Input: {floats}")
    
    float_tree = BinaryTree.from_sorted_values(floats)
    float_result = float_tree.traverse_inorder()
    
    print(f"   Sorted: {float_result}")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
