"""
Tree Implementation

A tree is a connected, acyclic undirected graph.
A forest is an acyclic undirected graph (can contain multiple trees).

Properties of a tree:
- For m edges and n nodes: m = n - 1
- Each pair of nodes is connected by exactly one path
- Connected, but becomes disconnected if an edge is removed
- Acyclic, but becomes cyclic if an edge is added

Terms:
- Root: distinguished node
- Parent (predecessor): direct ancestor
- Child (successor): direct descendant
- Leaf: node without children
- Inner nodes: nodes with children
- Degree: number of children of a node
- Depth: length of the path from root to node
- Level: all nodes of the same depth
- Height: maximum depth of any node
"""

from typing import Any, List, Optional, Set, Dict, TYPE_CHECKING
from collections import deque
import networkx as nx
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from .binary_tree import BinaryTree, BinaryNode


class Node:
    """
    Node of a tree (tree_node).
    
    Attributes:
        data: Data stored in the node
        parent: Pointer to parent node (predecessor)
        children: List of pointers to child nodes (successors)
    """
    
    def __init__(self, data: Any):
        """
        Creates a new tree node.
        
        Args:
            data: The data to store
        """
        self.data = data
        self.parent: Optional['Node'] = None
        self.children: List['Node'] = []
    
    def add_child(self, child: 'Node') -> None:
        """Adds a child node."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
    
    def remove_child(self, child: 'Node') -> bool:
        """
        Removes a child node.
        
        Returns:
            True if successful, False if child not found
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            return True
        return False
    
    def is_leaf(self) -> bool:
        """Checks if node is a leaf (has no children)."""
        return len(self.children) == 0
    
    def is_inner_node(self) -> bool:
        """Checks if node is an inner node (has children)."""
        return len(self.children) > 0
    
    def degree(self) -> int:
        """
        Returns the degree of the node (number of children).
        
        Note: Degree in a tree is defined differently than in a graph!
        In tree: number of children
        In graph: number of connected edges
        """
        return len(self.children)
    
    def __str__(self) -> str:
        return f"Node({self.data})"
    
    def __repr__(self) -> str:
        return f"Node(data={self.data}, children={len(self.children)})"


class Tree:
    """
    Tree Implementation.
    
    A tree is a connected, acyclic undirected graph
    with one distinguished node as the root.
    
    Properties:
    - m = n - 1 (edges = nodes - 1)
    - Exactly one path between each pair of nodes
    - Connected and acyclic
    - Each node (except root) has exactly one parent
    """
    
    def __init__(self, root_data: Any = None):
        """
        Creates a new tree.
        
        Args:
            root_data: Data for the root (optional)
        """
        self.root: Optional[Node] = Node(root_data) if root_data is not None else None
    
    def is_empty(self) -> bool:
        """Checks if tree is empty."""
        return self.root is None
    
    def set_root(self, data: Any) -> Node:
        """
        Sets the root of the tree.
        
        Args:
            data: Data for the root
            
        Returns:
            The created root node
        """
        self.root = Node(data)
        return self.root
    
    def add_child(self, parent: Node, child_data: Any) -> Node:
        """
        Adds a child to a parent node.
        
        Args:
            parent: Parent node
            child_data: Data for the new child
            
        Returns:
            The created child node
        """
        child = Node(child_data)
        parent.add_child(child)
        return child
    
    @classmethod
    def from_adjacency_matrix(cls, matrix: List[List[int]], node_labels: Optional[List[Any]] = None) -> 'Tree':
        """
        Creates a tree from an adjacency matrix.
        
        The adjacency matrix represents parent-child relationships where
        matrix[i][j] = 1 means node i is the parent of node j.
        
        Args:
            matrix: Adjacency matrix (2D list) where matrix[i][j] = 1 if i is parent of j
            node_labels: Optional list of labels for nodes (default: 0, 1, 2, ...)
            
        Returns:
            A new Tree instance
            
        Raises:
            ValueError: If matrix doesn't represent a valid tree structure
            
        Example:
            >>> # Tree structure: 0 -> 1, 0 -> 2, 1 -> 3
            >>> matrix = [
            ...     [0, 1, 1, 0],  # Node 0 has children 1, 2
            ...     [0, 0, 0, 1],  # Node 1 has child 3
            ...     [0, 0, 0, 0],  # Node 2 has no children
            ...     [0, 0, 0, 0]   # Node 3 has no children
            ... ]
            >>> tree = Tree.from_adjacency_matrix(matrix)
        """
        n = len(matrix)
        if n == 0:
            return cls()
        
        # Validate matrix dimensions
        if not all(len(row) == n for row in matrix):
            raise ValueError("Adjacency matrix must be square")
        
        # Use node labels or default to indices
        if node_labels is None:
            node_labels = list(range(n))
        elif len(node_labels) != n:
            raise ValueError(f"node_labels length ({len(node_labels)}) must match matrix size ({n})")
        
        # Find the root (node with no parent)
        parent_count = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
        root_candidates = [i for i, count in enumerate(parent_count) if count == 0]
        
        if len(root_candidates) == 0:
            raise ValueError("No root found - tree must have exactly one node with no parent")
        if len(root_candidates) > 1:
            raise ValueError(f"Multiple roots found at indices {root_candidates} - tree must have exactly one root")
        
        root_idx = root_candidates[0]
        
        # Verify each non-root node has exactly one parent
        for j in range(n):
            if j != root_idx and parent_count[j] != 1:
                raise ValueError(f"Node {j} has {parent_count[j]} parents - each node must have exactly one parent")
        
        # Create tree with root
        tree = cls(node_labels[root_idx])
        
        # Create a mapping from index to Node
        nodes = {root_idx: tree.root}
        
        # Build tree using BFS to ensure parents are created before children
        queue = deque([root_idx])
        visited = {root_idx}
        
        while queue:
            parent_idx = queue.popleft()
            parent_node = nodes[parent_idx]
            
            # Find all children of current parent
            for child_idx in range(n):
                if matrix[parent_idx][child_idx] == 1 and child_idx not in visited:
                    # Create child node
                    child_node = tree.add_child(parent_node, node_labels[child_idx])
                    nodes[child_idx] = child_node
                    visited.add(child_idx)
                    queue.append(child_idx)
        
        # Verify all nodes were visited
        if len(visited) != n:
            unvisited = set(range(n)) - visited
            raise ValueError(f"Not all nodes reachable from root - disconnected nodes: {unvisited}")
        
        return tree
    
    @classmethod
    def from_nested_structure(cls, structure: Any) -> 'Tree':
        """
        Creates a tree from a nested structure (tuple/list format).
        
        This method allows creating trees with duplicate node values by using
        a nested structure where each node is represented as either:
        - A single value (for leaf nodes)
        - A tuple/list: (value, [child1, child2, ...]) (for nodes with children)
        
        Args:
            structure: Nested structure representing the tree
            
        Returns:
            A new Tree instance
            
        Example:
            >>> # Tree with duplicate values: + has two * children, each * has different children
            >>> structure = ('+', [
            ...     ('*', [
            ...         ('+', [3, 4]),
            ...         5
            ...     ]),
            ...     ('*', [2, 3])
            ... ])
            >>> tree = Tree.from_nested_structure(structure)
        """
        def build_tree_recursive(parent_node: Optional[Node], node_structure: Any, tree: 'Tree') -> Node:
            """Helper function to recursively build tree from structure."""
            # Check if node has children (tuple/list with 2 elements)
            if isinstance(node_structure, (tuple, list)) and len(node_structure) == 2:
                value, children = node_structure
                
                # Create or use node
                if parent_node is None:
                    # Root node
                    tree.set_root(value)
                    current_node = tree.root
                else:
                    # Child node
                    current_node = tree.add_child(parent_node, value)
                
                # Recursively add children
                if isinstance(children, list):
                    for child_structure in children:
                        build_tree_recursive(current_node, child_structure, tree)
                
                return current_node
            else:
                # Leaf node (just a value)
                if parent_node is None:
                    tree.set_root(node_structure)
                    return tree.root
                else:
                    return tree.add_child(parent_node, node_structure)
        
        tree = cls()
        build_tree_recursive(None, structure, tree)
        return tree
    
    @classmethod
    def from_adjacency_list(cls, adj_list: Dict[Any, List[Any]], root: Any) -> 'Tree':
        """
        Creates a tree from an adjacency list.
        
        The adjacency list is a dictionary where each key is a parent node
        and the value is a list of its children.
        
        Note: This method assumes each node value is unique. For trees with
        duplicate node values, use from_nested_structure() instead.
        
        Args:
            adj_list: Dictionary mapping parent nodes to lists of children
            root: The root node value
            
        Returns:
            A new Tree instance
            
        Raises:
            ValueError: If the structure doesn't represent a valid tree
            
        Example:
            >>> adj_list = {
            ...     'A': ['B', 'C'],
            ...     'B': ['D'],
            ...     'C': [],
            ...     'D': []
            ... }
            >>> tree = Tree.from_adjacency_list(adj_list, 'A')
        """
        if root not in adj_list:
            raise ValueError(f"Root '{root}' not found in adjacency list")
        
        # Create tree with root
        tree = cls(root)
        
        # Track nodes to detect cycles and ensure tree property
        nodes = {root: tree.root}
        visited = {root}
        
        # Build tree using BFS
        queue = deque([root])
        
        while queue:
            parent_value = queue.popleft()
            parent_node = nodes[parent_value]
            
            # Get children of current parent
            children = adj_list.get(parent_value, [])
            
            for child_value in children:
                # Check for cycles
                if child_value in visited:
                    raise ValueError(f"Cycle detected: node '{child_value}' appears multiple times")
                
                # Create child node
                child_node = tree.add_child(parent_node, child_value)
                nodes[child_value] = child_node
                visited.add(child_value)
                queue.append(child_value)
        
        return tree
    
    def get_node_count(self) -> int:
        """
        Returns the number of nodes in the tree (n).
        
        Returns:
            Number of nodes
        """
        if self.is_empty():
            return 0
        return self._count_nodes(self.root)
    
    def _count_nodes(self, node: Node) -> int:
        """Helper method to count nodes (recursive)."""
        count = 1  # Current node
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def get_edge_count(self) -> int:
        """
        Returns the number of edges in the tree (m).
        
        For trees: m = n - 1
        
        Returns:
            Number of edges
        """
        n = self.get_node_count()
        return n - 1 if n > 0 else 0
    
    def verify_tree_property(self) -> bool:
        """
        Verifies the tree property: m = n - 1
        
        Returns:
            True if the property is satisfied
        """
        n = self.get_node_count()
        m = self.get_edge_count()
        return m == n - 1
    
    def get_depth(self, node: Node) -> int:
        """
        Returns the depth of a node.
        
        Depth = length of the path from root to node
        
        Args:
            node: The node
            
        Returns:
            Depth of the node
        """
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def get_degree(self, node: Node) -> int:
        """
        Returns the degree of a node (number of children).
        
        The degree of a node is the number of children it has.
        
        Args:
            node: The node
            
        Returns:
            Number of children the node has
            
        Example:
            >>> tree = Tree("Root")
            >>> child_a = tree.add_child(tree.root, "A")
            >>> tree.add_child(tree.root, "B")
            >>> tree.get_degree(tree.root)  # 2 children
            2
            >>> tree.get_degree(child_a)    # 0 children (leaf)
            0
        """
        return len(node.children)
    
    def get_height(self) -> int:
        """
        Returns the height of the tree.
        
        Height = maximum depth of any node
        
        Returns:
            Height of the tree
        """
        if self.is_empty():
            return -1
        return self._get_height(self.root)
    
    def _get_height(self, node: Node) -> int:
        """Helper method to calculate height (recursive)."""
        if node.is_leaf():
            return 0
        max_height = 0
        for child in node.children:
            child_height = self._get_height(child)
            max_height = max(max_height, child_height)
        return max_height + 1
    
    def get_level(self, depth: int) -> List[Node]:
        """
        Returns all nodes at a specific level.
        
        Level = all nodes with the same depth
        
        Args:
            depth: The desired depth
            
        Returns:
            List of all nodes at this level
        """
        if self.is_empty():
            return []
        
        nodes_at_level = []
        self._collect_level_nodes(self.root, 0, depth, nodes_at_level)
        return nodes_at_level
    
    def _collect_level_nodes(self, node: Node, current_depth: int, 
                            target_depth: int, result: List[Node]) -> None:
        """Helper method to collect nodes at a level."""
        if current_depth == target_depth:
            result.append(node)
            return
        
        for child in node.children:
            self._collect_level_nodes(child, current_depth + 1, target_depth, result)
    
    def get_all_levels(self) -> Dict[int, List[Node]]:
        """
        Returns all levels of the tree.
        
        Returns:
            Dictionary: {depth: [nodes at this level]}
        """
        levels = {}
        if not self.is_empty():
            self._collect_all_levels(self.root, 0, levels)
        return levels
    
    def _collect_all_levels(self, node: Node, depth: int, 
                           levels: Dict[int, List[Node]]) -> None:
        """Helper method to collect all levels."""
        if depth not in levels:
            levels[depth] = []
        levels[depth].append(node)
        
        for child in node.children:
            self._collect_all_levels(child, depth + 1, levels)
    
    def get_leaves(self) -> List[Node]:
        """
        Returns all leaves of the tree.
        
        Leaf = node without children
        
        Returns:
            List of all leaves
        """
        if self.is_empty():
            return []
        
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves
    
    def _collect_leaves(self, node: Node, leaves: List[Node]) -> None:
        """Helper method to collect leaves."""
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)
    
    def get_inner_nodes(self) -> List[Node]:
        """
        Returns all inner nodes of the tree.
        
        Inner node = node with at least one child
        
        Returns:
            List of all inner nodes
        """
        if self.is_empty():
            return []
        
        inner_nodes = []
        self._collect_inner_nodes(self.root, inner_nodes)
        return inner_nodes
    
    def _collect_inner_nodes(self, node: Node, inner_nodes: List[Node]) -> None:
        """Helper method to collect inner nodes."""
        if node.is_inner_node():
            inner_nodes.append(node)
            for child in node.children:
                self._collect_inner_nodes(child, inner_nodes)
    
    def find_path(self, from_node: Node, to_node: Node) -> Optional[List[Node]]:
        """
        Finds the unique path between two nodes.
        
        In a tree, there is exactly one path between each pair of nodes.
        
        Args:
            from_node: Start node
            to_node: Target node
            
        Returns:
            List of nodes on the path (including start and target)
            or None if nodes are not connected
        """
        # Find path from from_node to root
        path_to_root_from = []
        current = from_node
        while current is not None:
            path_to_root_from.append(current)
            current = current.parent
        
        # Find path from to_node to root
        path_to_root_to = []
        current = to_node
        while current is not None:
            path_to_root_to.append(current)
            current = current.parent
        
        # Find lowest common ancestor
        common_ancestor = None
        for node_from in path_to_root_from:
            if node_from in path_to_root_to:
                common_ancestor = node_from
                break
        
        if common_ancestor is None:
            return None  # Nodes not connected (different trees)
        
        # Construct path: from_node -> common_ancestor -> to_node
        path = []
        
        # Part 1: from_node to common_ancestor
        current = from_node
        while current != common_ancestor:
            path.append(current)
            current = current.parent
        path.append(common_ancestor)
        
        # Part 2: common_ancestor to to_node
        path_from_ancestor = []
        current = to_node
        while current != common_ancestor:
            path_from_ancestor.append(current)
            current = current.parent
        
        # Reverse and append
        path_from_ancestor.reverse()
        path.extend(path_from_ancestor)
        
        return path
    
    def is_connected(self) -> bool:
        """
        Checks if the tree is connected.
        
        A valid tree is always connected.
        
        Returns:
            True if connected
        """
        if self.is_empty():
            return True
        
        # All nodes reachable from root?
        visited = set()
        self._visit_all(self.root, visited)
        return len(visited) == self.get_node_count()
    
    def _visit_all(self, node: Node, visited: Set[Node]) -> None:
        """Helper method to visit all nodes."""
        visited.add(node)
        for child in node.children:
            self._visit_all(child, visited)
    
    def has_cycle(self) -> bool:
        """
        Checks if the tree contains a cycle.
        
        A valid tree is always acyclic.
        
        Returns:
            False (tree is always acyclic)
        """
        # A correctly constructed tree cannot have cycles
        # This method exists for consistency with graph theory
        return False
    
    def is_acyclic(self) -> bool:
        """
        Checks if the tree is acyclic.
        
        A valid tree is always acyclic.
        
        Returns:
            True (tree is always acyclic)
        """
        return True
    
    def traverse_preorder(self, node: Optional[Node] = None) -> List[Any]:
        """
        Traverses the tree in preorder.
        
        Order: Root -> Children (left to right)
        
        Args:
            node: Start node (default: root)
            
        Returns:
            List of data in preorder
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        result = [node.data]
        for child in node.children:
            result.extend(self.traverse_preorder(child))
        return result
    
    def traverse_postorder(self, node: Optional[Node] = None) -> List[Any]:
        """
        Traverses the tree in postorder.
        
        Order: Children (left to right) -> Root
        
        Args:
            node: Start node (default: root)
            
        Returns:
            List of data in postorder
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        result = []
        for child in node.children:
            result.extend(self.traverse_postorder(child))
        result.append(node.data)
        return result
    
    def traverse_levelorder(self) -> List[Any]:
        """
        Traverses the tree in level-order (breadth-first).
        
        Order: Level by level from top to bottom
        
        Returns:
            List of data in level-order
        """
        if self.is_empty():
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            result.append(node.data)
            queue.extend(node.children)
        
        return result
    
    def traverse_inorder(self, node: Optional[Node] = None) -> List[Any]:
        """
        Traverses the tree in inorder.
        
        For general trees (not binary): Left subtree -> Root -> Right subtrees
        This implementation visits the first child, then the node, then remaining children.
        
        Args:
            node: Start node (default: root)
            
        Returns:
            List of data in inorder
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        result = []
        
        # For general trees, we can define inorder as:
        # Process first child subtree, then root, then remaining children
        if node.children:
            # Process first child
            result.extend(self.traverse_inorder(node.children[0]))
            
            # Process root
            result.append(node.data)
            
            # Process remaining children
            for child in node.children[1:]:
                result.extend(self.traverse_inorder(child))
        else:
            # Leaf node
            result.append(node.data)
        
        return result
    
    def find_node(self, data: Any) -> Optional[Node]:
        """
        Finds a node with specific data.
        
        Args:
            data: The data to search for
            
        Returns:
            The found node or None
        """
        if self.is_empty():
            return None
        
        return self._find_node_recursive(self.root, data)
    
    def _find_node_recursive(self, node: Node, data: Any) -> Optional[Node]:
        """Helper method for node search."""
        if node.data == data:
            return node
        
        for child in node.children:
            result = self._find_node_recursive(child, data)
            if result is not None:
                return result
        
        return None
    
    def get_ancestors(self, node: Node) -> List[Node]:
        """
        Returns all ancestors of a node.
        
        Args:
            node: The node
            
        Returns:
            List of all ancestors (from parent to root)
        """
        ancestors = []
        current = node.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_descendants(self, node: Node) -> List[Node]:
        """
        Returns all descendants of a node.
        
        Args:
            node: The node
            
        Returns:
            List of all descendants
        """
        descendants = []
        self._collect_descendants(node, descendants)
        return descendants
    
    def _collect_descendants(self, node: Node, descendants: List[Node]) -> None:
        """Helper method to collect descendants."""
        for child in node.children:
            descendants.append(child)
            self._collect_descendants(child, descendants)
    
    def print_tree(self, node: Optional[Node] = None, prefix: str = "", is_last: bool = True) -> None:
        """
        Prints the tree in a visual representation.
        
        Args:
            node: Start node (default: root)
            prefix: Prefix for indentation (internal)
            is_last: Whether node is last in list (internal)
        """
        if node is None:
            node = self.root
        
        if node is None:
            print("Empty tree")
            return
        
        # Print current node
        connector = "└── " if is_last else "├── "
        print(prefix + connector + str(node.data))
        
        # New prefix for children
        extension = "    " if is_last else "│   "
        new_prefix = prefix + extension
        
        # Print children
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            self.print_tree(child, new_prefix, is_last_child)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the tree.
        
        Returns:
            Dictionary with various statistics
        """
        if self.is_empty():
            return {
                'node_count': 0,
                'edge_count': 0,
                'height': -1,
                'leaf_count': 0,
                'inner_node_count': 0,
                'satisfies_tree_property': True
            }
        
        leaves = self.get_leaves()
        inner_nodes = self.get_inner_nodes()
        
        return {
            'node_count': self.get_node_count(),
            'edge_count': self.get_edge_count(),
            'height': self.get_height(),
            'leaf_count': len(leaves),
            'inner_node_count': len(inner_nodes),
            'satisfies_tree_property': self.verify_tree_property(),
            'is_connected': self.is_connected(),
            'is_acyclic': self.is_acyclic()
        }
    
    def get_adjacency_matrix(self) -> List[List[int]]:
        """
        Returns the adjacency matrix representation of the tree.
        
        The adjacency matrix is a 2D list where matrix[i][j] = 1 if node i
        is the parent of node j, and 0 otherwise.
        
        Returns:
            2D list representing the adjacency matrix
            
        Example:
            >>> tree = Tree("A")
            >>> tree.add_child(tree.root, "B")
            >>> tree.add_child(tree.root, "C")
            >>> matrix = tree.get_adjacency_matrix()
            >>> # matrix[0][1] = 1 (A is parent of B)
            >>> # matrix[0][2] = 1 (A is parent of C)
        """
        if self.is_empty():
            return []
        
        # Get all nodes in level-order (BFS)
        nodes = []
        node_to_idx = {}
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            node_to_idx[id(node)] = len(nodes)
            nodes.append(node)
            queue.extend(node.children)
        
        n = len(nodes)
        matrix = [[0] * n for _ in range(n)]
        
        # Fill matrix: matrix[i][j] = 1 if node i is parent of node j
        for i, parent_node in enumerate(nodes):
            for child_node in parent_node.children:
                j = node_to_idx[id(child_node)]
                matrix[i][j] = 1
        
        return matrix
    
    def get_adjacency_list(self) -> Dict[Any, List[Any]]:
        """
        Returns the adjacency list representation of the tree.
        
        The adjacency list is a dictionary where each key is a node's data
        and the value is a list of its children's data.
        
        Returns:
            Dictionary mapping node data to list of children data
            
        Example:
            >>> tree = Tree("A")
            >>> tree.add_child(tree.root, "B")
            >>> tree.add_child(tree.root, "C")
            >>> adj_list = tree.get_adjacency_list()
            >>> # {'A': ['B', 'C'], 'B': [], 'C': []}
        """
        if self.is_empty():
            return {}
        
        adj_list = {}
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            # Store children data
            adj_list[node.data] = [child.data for child in node.children]
            queue.extend(node.children)
        
        return adj_list
    
    def to_nested_structure(self, node: Optional[Node] = None) -> Any:
        """
        Converts the tree to a nested structure representation.
        
        This method converts a tree into the nested tuple/list format used by
        from_nested_structure(). This is useful for trees with duplicate node values,
        as it preserves the exact structure without ambiguity.
        
        Args:
            node: The node to start from (default: root). Used for recursion.
            
        Returns:
            Nested structure where each node is either:
            - A single value (for leaf nodes)
            - A tuple (value, [children]) (for nodes with children)
            
        Example:
            >>> tree = Tree("+")
            >>> left = tree.add_child(tree.root, "*")
            >>> tree.add_child(left, 3)
            >>> tree.add_child(left, 4)
            >>> right = tree.add_child(tree.root, "*")
            >>> tree.add_child(right, 5)
            >>> structure = tree.to_nested_structure()
            >>> # ('+', [('*', [3, 4]), ('*', [5])])
        """
        if self.is_empty():
            return None
        
        if node is None:
            node = self.root
        
        # If leaf node, return just the value
        if node.is_leaf():
            return node.data
        
        # If node has children, return (value, [children])
        children = [self.to_nested_structure(child) for child in node.children]
        return (node.data, children)
    
    def get_node_labels(self) -> List[Any]:
        """
        Returns the node labels in the same order as the adjacency matrix.
        
        This method returns node labels in BFS (level-order) traversal, which matches
        the order used by get_adjacency_matrix(). You can use this with 
        Tree.from_adjacency_matrix() to reconstruct the tree.
        
        Returns:
            List of node labels in BFS order
            
        Example:
            >>> tree = Tree("A")
            >>> b = tree.add_child(tree.root, "B")
            >>> c = tree.add_child(tree.root, "C")
            >>> tree.add_child(b, "D")
            >>> labels = tree.get_node_labels()
            >>> # ['A', 'B', 'C', 'D']
            >>> matrix = tree.get_adjacency_matrix()
            >>> tree2 = Tree.from_adjacency_matrix(matrix, labels)
            >>> # tree2 is identical to tree
        """
        if self.is_empty():
            return []
        
        # Get all nodes in level-order (BFS) - same order as adjacency matrix
        labels = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            labels.append(node.data)
            queue.extend(node.children)
        
        return labels
    
    def visualize(self, title: str = "", figsize: tuple = (12, 9), 
                  root_position: str = "top", positions: Optional[Dict[Any, tuple]] = None):
        """
        Visualize the tree using matplotlib and networkx.
        
        Args:
            title: Title for the tree visualization
            figsize: Figure size as (width, height)
            root_position: Position of root - "top", "bottom", "left", or "right"
            positions: Optional dictionary mapping nodes to (x, y) coordinates
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib and networkx are required for visualization")
            print("Install with: pip install matplotlib networkx")
            return
        
        if self.is_empty():
            print("Tree is empty - nothing to visualize")
            return
        
        # Create NetworkX directed graph (tree structure)
        G = nx.DiGraph()
        
        # Create mappings for unique node identification
        node_to_id = {}  # Node -> unique int ID
        id_to_data = {}  # unique int ID -> data value
        
        # Add all nodes and edges using unique IDs
        self._add_nodes_to_graph(G, self.root, node_to_id, id_to_data)
        
        # Calculate maximum width at any level for dynamic spacing
        levels = self.get_all_levels()
        max_width = max(len(nodes) for nodes in levels.values()) if levels else 1
        
        # Count total leaves for better width estimation
        leaf_count = len(self.get_leaves())
        node_count = self.get_node_count()
        
        # Adaptive figure sizing - compact for small trees, spacious for large trees
        height = self.get_height() + 1
        
        # Adaptive width scaling: small trees are compact, large trees get more space
        if leaf_count <= 5:
            width_factor = 1.2  # Very compact for tiny trees
        elif leaf_count <= 15:
            width_factor = 1.5  # Compact for small trees
        elif leaf_count <= 30:
            width_factor = 1.8  # Moderate for medium trees
        else:
            width_factor = 2.2  # More space for large trees
        
        dynamic_width = max(8, leaf_count * width_factor, max_width * 2)
        
        # Adaptive height scaling
        if height <= 3:
            height_factor = 2.5  # Compact for shallow trees
        elif height <= 6:
            height_factor = 3.0  # Moderate for medium depth
        else:
            height_factor = 3.5  # More space for deep trees
        
        dynamic_height = max(6, height * height_factor)
        plt.figure(figsize=(dynamic_width, dynamic_height))
        
        if positions:
            pos = positions
        else:
            # Use hierarchical layout based on root position
            root_id = node_to_id[self.root]
            if root_position == "top":
                pos = self._get_hierarchical_pos(G, root_id, max_width, orientation="vertical", reverse=False)
            elif root_position == "bottom":
                pos = self._get_hierarchical_pos(G, root_id, max_width, orientation="vertical", reverse=True)
            elif root_position == "left":
                pos = self._get_hierarchical_pos(G, root_id, max_width, orientation="horizontal", reverse=True)
            elif root_position == "right":
                pos = self._get_hierarchical_pos(G, root_id, max_width, orientation="horizontal", reverse=False)
            else:
                # Fallback to spring layout
                pos = nx.spring_layout(G, seed=42)
        
        # Categorize nodes using unique IDs
        root_nodes = [node_to_id[self.root]]
        leaf_nodes = [node_to_id[leaf] for leaf in self.get_leaves()]
        inner_nodes = [node_to_id[node] for node in self.get_inner_nodes() if node != self.root]
        
        # Draw nodes with different colors
        if root_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=root_nodes,
                                  node_color='mediumpurple', node_size=1000,
                                  alpha=0.9)
        
        if inner_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=inner_nodes,
                                  node_color='orange', node_size=800,
                                  alpha=0.9)
        
        if leaf_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes,
                                  node_color='lightgreen', node_size=800,
                                  alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos,
                              edge_color='gray',
                              width=2,
                              arrows=True,
                              arrowsize=15,
                              arrowstyle='->')
        
        # Draw labels using actual data values (not IDs)
        labels = {node_id: str(data) for node_id, data in id_to_data.items()}
        nx.draw_networkx_labels(G, pos,
                               labels=labels,
                               font_size=10,
                               font_weight='bold',
                               font_color='black')
        
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Add legend
        legend_elements = [
            Patch(facecolor='mediumpurple', label='Root'),
            Patch(facecolor='orange', label='Inner Nodes'),
            Patch(facecolor='lightgreen', label='Leaves')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.show()

    def _add_nodes_to_graph(self, G, node: Node, node_to_id: Dict[Node, int], id_to_data: Dict[int, Any]):
        """Helper method to add nodes and edges to NetworkX graph using unique IDs."""
        node_id = id(node)  # Use object id as unique identifier
        node_to_id[node] = node_id
        id_to_data[node_id] = node.data
        G.add_node(node_id)
        for child in node.children:
            child_id = id(child)
            G.add_edge(node_id, child_id)
            self._add_nodes_to_graph(G, child, node_to_id, id_to_data)

    def _get_hierarchical_pos(self, G, root, max_width, orientation="vertical", reverse=False):
        """
        Create hierarchical layout for tree visualization that prevents subtree overlap.
        Uses a proper subtree-aware algorithm that calculates actual subtree widths.
        
        Args:
            G: NetworkX graph
            root: Root node ID
            max_width: Maximum number of nodes at any level
            orientation: "vertical" or "horizontal"
            reverse: Whether to reverse the direction
        """
        # Adaptive spacing based on tree complexity
        # Count total nodes for complexity estimation
        total_nodes = len(G.nodes())
        
        # Scale spacing with tree size - compact for small, spacious for large
        if total_nodes <= 10:
            min_spacing = 0.8   # Compact spacing for small trees
            vert_gap = 0.8
        elif total_nodes <= 30:
            min_spacing = 1.2   # Moderate spacing for medium trees
            vert_gap = 1.0
        elif total_nodes <= 60:
            min_spacing = 1.6   # More spacing for larger trees
            vert_gap = 1.1
        else:
            min_spacing = 2.0   # Generous spacing for very large trees
            vert_gap = 1.2
        
        # Calculate subtree widths (how many leaf nodes each subtree has)
        subtree_widths = {}
        
        def calculate_subtree_width(node, parent=None):
            """Calculate the width needed for a subtree (leaf count)."""
            children = [n for n in G.neighbors(node) if n != parent]
            
            if not children:
                # Leaf node needs width of 1
                subtree_widths[node] = 1.0
                return 1.0
            
            # Internal node: sum of children's widths plus spacing
            total_width = 0.0
            for child in children:
                child_width = calculate_subtree_width(child, node)
                total_width += child_width
            
            # Add spacing between children's subtrees
            total_width += (len(children) - 1) * min_spacing
            subtree_widths[node] = max(total_width, 1.0)
            return subtree_widths[node]
        
        # Calculate widths for all subtrees
        total_width = calculate_subtree_width(root)
        
        # Position nodes using calculated widths
        pos = {}
        
        def position_subtree(node, x_center, y_pos, parent=None):
            """Position a subtree centered at x_center."""
            pos[node] = (x_center, y_pos)
            
            children = [n for n in G.neighbors(node) if n != parent]
            if not children:
                return
            
            # Calculate total width needed for all children
            children_widths = [subtree_widths[child] for child in children]
            total_children_width = sum(children_widths) + (len(children) - 1) * min_spacing
            
            # Start position for first child (left edge of children group)
            x_current = x_center - total_children_width / 2
            
            # Position each child
            for i, child in enumerate(children):
                child_width = children_widths[i]
                # Center the child in its allocated width
                child_x = x_current + child_width / 2
                position_subtree(child, child_x, y_pos - vert_gap, node)
                # Move to next child's position
                x_current += child_width + min_spacing
        
        # Start positioning from root at center
        position_subtree(root, 0, 0)
        
        pos = pos
        
        if orientation == "horizontal":
            # Swap x and y coordinates
            pos = {node: (y, x) for node, (x, y) in pos.items()}
        
        if reverse:
            if orientation == "vertical":
                # Flip vertically
                max_y = max(y for x, y in pos.values())
                pos = {node: (x, max_y - y) for node, (x, y) in pos.items()}
            else:
                # Flip horizontally  
                max_x = max(x for x, y in pos.values())
                pos = {node: (max_x - x, y) for node, (x, y) in pos.items()}
        
        return pos
    
    def to_binary_tree(self, preserve_binary: bool = False) -> 'BinaryTree':
        """
        Convert this tree to a BinaryTree using LCRS (Left-Child Right-Sibling) representation.
        
        In LCRS representation:
        - Left child pointer points to first child of the node
        - Right child pointer points to next sibling of the node
        
        This allows representing any general tree as a binary tree.
        
        Args:
            preserve_binary: If True, preserve subtrees that are already binary (max 2 children).
                           Only apply LCRS conversion to nodes with 3+ children.
                           If False (default), apply LCRS consistently to all nodes.
        
        Returns:
            A new BinaryTree with LCRS structure
            
        Example:
            Original tree:
                  A
                / | \\
               B  C  D
              / \\
             E   F
            
            LCRS Binary tree (preserve_binary=False):
                  A
                 /
                B
               / \\
              E   C
               \\   \\
                F   D
            
            With preserve_binary=True, subtrees with ≤2 children maintain structure.
        """
        from .binary_tree import BinaryTree, BinaryNode
        
        if self.root is None:
            return BinaryTree()
        
        binary_tree = BinaryTree(self.root.data)
        self._convert_to_binary_recursive(self.root, binary_tree.root, preserve_binary)
        return binary_tree
    
    def _convert_to_binary_recursive(self, tree_node: Node, binary_node: 'BinaryNode', preserve_binary: bool = False) -> None:
        """
        Recursive helper to convert tree structure to LCRS binary tree.
        
        Args:
            tree_node: Current node in original tree
            binary_node: Corresponding node in binary tree
            preserve_binary: Whether to preserve binary subtrees when possible
            
        Note:
            When preserve_binary=True:
            - Nodes with ≤2 children: Attempt to preserve as left/right children
            - Nodes with >2 children: Always use LCRS
            
            However, preservation can only succeed if the node's right pointer is available.
            If a parent uses LCRS (has >2 children), child nodes become siblings and their
            right pointers are used for the sibling chain, preventing binary preservation.
        """
        from .binary_tree import BinaryNode
        
        children = tree_node.children
        if not children:
            return
        
        # Check if binary_node.right is already claimed (by sibling chain)
        right_available = (binary_node.right is None)
        
        # Decide whether to preserve or use LCRS for this node's children
        if preserve_binary and len(children) <= 2:
            # Attempt to preserve binary structure
            first_child = BinaryNode(children[0].data)
            binary_node.left = first_child
            first_child.parent = binary_node
            self._convert_to_binary_recursive(children[0], first_child, preserve_binary)
            
            if len(children) == 2:
                second_child = BinaryNode(children[1].data)
                if right_available:
                    # Right pointer available, preserve as right child
                    binary_node.right = second_child
                    second_child.parent = binary_node
                else:
                    # Right pointer not available (used by sibling), chain as sibling of first_child
                    first_child.right = second_child
                    second_child.parent = binary_node
                self._convert_to_binary_recursive(children[1], second_child, preserve_binary)
        else:
            # Use LCRS: left = first child, siblings form a chain via right pointers
            first_child = BinaryNode(children[0].data)
            binary_node.left = first_child
            first_child.parent = binary_node
            
            # Create sibling chain first
            current_sibling = first_child
            for i in range(1, len(children)):
                next_sibling = BinaryNode(children[i].data)
                current_sibling.right = next_sibling
                next_sibling.parent = binary_node
                current_sibling = next_sibling
            
            # Now recurse into each child
            # Since right pointers are already set for siblings, preservation will be limited
            current = first_child
            for child in children:
                self._convert_to_binary_recursive(child, current, preserve_binary)
                current = current.right
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        return (f"Tree(nodes={stats['node_count']}, "
                f"edges={stats['edge_count']}, "
                f"height={stats['height']})")
    
    def __repr__(self) -> str:
        return self.__str__()


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EXAMPLE 1: Create Tree Manually")
    print("=" * 80)
    
    # Create tree
    tree = Tree("Root")
    
    # Add children to root
    child_a = tree.add_child(tree.root, "A")
    child_b = tree.add_child(tree.root, "B")
    child_c = tree.add_child(tree.root, "C")
    
    # Add children to A
    child_a1 = tree.add_child(child_a, "A1")
    child_a2 = tree.add_child(child_a, "A2")
    
    # Add children to B
    child_b1 = tree.add_child(child_b, "B1")
    
    # Visualization
    print("=== Tree Structure ===")
    tree.print_tree()
    
    print("\n=== Statistics ===")
    stats = tree.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n=== Tree Property (m = n - 1) ===")
    print(f"Nodes (n): {tree.get_node_count()}")
    print(f"Edges (m): {tree.get_edge_count()}")
    print(f"m = n - 1? {tree.verify_tree_property()}")
    
    print("\n=== Leaves ===")
    leaves = tree.get_leaves()
    print([leaf.data for leaf in leaves])
    
    print("\n=== Inner Nodes ===")
    inner = tree.get_inner_nodes()
    print([node.data for node in inner])
    
    print("\n=== Levels ===")
    levels = tree.get_all_levels()
    for depth, nodes in levels.items():
        print(f"Level {depth}: {[node.data for node in nodes]}")
    
    print("\n=== Traversals ===")
    print(f"Preorder: {tree.traverse_preorder()}")
    print(f"In-Order: {tree.traverse_inorder()}")
    print(f"Postorder: {tree.traverse_postorder()}")
    print(f"Level-Order: {tree.traverse_levelorder()}")
    
    
    print("\n=== Path Between Nodes ===")
    path = tree.find_path(child_a2, child_b1)
    if path:
        print(f"Path from A2 to B1: {[node.data for node in path]}")
    
    print(f"\n=== Depth of Node A2 ===")
    print(f"Depth: {tree.get_depth(child_a2)}")
    
    print(f"\n=== Tree Height ===")
    print(f"Height: {tree.get_height()}")
    
    # Visualize tree with root at the top
    print("\n=== Tree Visualization (Root at Top) ===")
    tree.visualize(title="Tree with Root at Top", root_position="top")

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Create Tree from Adjacency Matrix")
    print("=" * 80)
    
    # Define adjacency matrix
    # Tree structure: 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
    matrix = [
        [0, 1, 1, 0, 0],  # Node 0 (root) has children 1, 2
        [0, 0, 0, 1, 1],  # Node 1 has children 3, 4
        [0, 0, 0, 0, 0],  # Node 2 has no children (leaf)
        [0, 0, 0, 0, 0],  # Node 3 has no children (leaf)
        [0, 0, 0, 0, 0]   # Node 4 has no children (leaf)
    ]
    
    # Optional: provide custom node labels
    labels = ['Root', 'A', 'B', 'A1', 'A2']
    
    tree2 = Tree.from_adjacency_matrix(matrix, labels)
    
    print("\nAdjacency Matrix:")
    for row in matrix:
        print(row)
    
    print("\nTree Structure:")
    tree2.print_tree()
    
    print("\nStatistics:")
    stats2 = tree2.get_statistics()
    print(f"Nodes: {stats2['node_count']}")
    print(f"Edges: {stats2['edge_count']}")
    print(f"Height: {stats2['height']}")
    print(f"Tree property (m = n - 1): {tree2.verify_tree_property()}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Create Tree from Adjacency List")
    print("=" * 80)
    
    # Define adjacency list
    adj_list = {
        'Root': ['A', 'B', 'C'],
        'A': ['A1', 'A2'],
        'B': ['B1'],
        'C': [],
        'A1': [],
        'A2': [],
        'B1': []
    }
    
    tree3 = Tree.from_adjacency_list(adj_list, 'Root')
    
    print("\nAdjacency List:")
    for parent, children in adj_list.items():
        print(f"  {parent}: {children}")
    
    print("\nTree Structure:")
    tree3.print_tree()
    
    print("\nStatistics:")
    stats3 = tree3.get_statistics()
    print(f"Nodes: {stats3['node_count']}")
    print(f"Edges: {stats3['edge_count']}")
    print(f"Height: {stats3['height']}")
    
    print("\nTraversals:")
    print(f"Preorder: {tree3.traverse_preorder()}")
    print(f"Level-order: {tree3.traverse_levelorder()}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Export Tree to Adjacency Matrix")
    print("=" * 80)
    
    # Use tree from Example 1
    print("\nOriginal Tree Structure:")
    tree.print_tree()
    
    matrix_export = tree.get_adjacency_matrix()
    print("\nExported Adjacency Matrix:")
    for i, row in enumerate(matrix_export):
        print(f"  Node {i}: {row}")
    
    print(f"\nMatrix dimensions: {len(matrix_export)} x {len(matrix_export[0])}")
    print("Note: matrix[i][j] = 1 means node i is the parent of node j")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Export Tree to Adjacency List")
    print("=" * 80)
    
    # Use tree from Example 1
    print("\nOriginal Tree Structure:")
    tree.print_tree()
    
    adj_list_export = tree.get_adjacency_list()
    print("\nExported Adjacency List:")
    for parent, children in adj_list_export.items():
        print(f"  {parent}: {children}")
    
    print(f"\nTotal nodes in adjacency list: {len(adj_list_export)}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Round-trip Conversion (Matrix -> Tree -> Matrix)")
    print("=" * 80)
    
    # Create tree from matrix
    original_matrix = [
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    labels_rt = ['X', 'Y', 'Z', 'W']
    
    print("Original Matrix:")
    for row in original_matrix:
        print(f"  {row}")
    
    tree_rt = Tree.from_adjacency_matrix(original_matrix, labels_rt)
    print("\nTree Structure:")
    tree_rt.print_tree()
    
    exported_matrix = tree_rt.get_adjacency_matrix()
    print("\nExported Matrix:")
    for row in exported_matrix:
        print(f"  {row}")
    
    # Verify they match
    matrices_match = original_matrix == exported_matrix
    print(f"\nMatrices match: {matrices_match}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Round-trip Conversion (List -> Tree -> List)")
    print("=" * 80)
    
    # Create tree from adjacency list
    original_list = {
        'P': ['Q', 'R'],
        'Q': ['S'],
        'R': [],
        'S': []
    }
    
    print("Original Adjacency List:")
    for k, v in original_list.items():
        print(f"  {k}: {v}")
    
    tree_rt2 = Tree.from_adjacency_list(original_list, 'P')
    print("\nTree Structure:")
    tree_rt2.print_tree()
    
    exported_list = tree_rt2.get_adjacency_list()
    print("\nExported Adjacency List:")
    for k, v in exported_list.items():
        print(f"  {k}: {v}")
    
    # Verify they match
    lists_match = original_list == exported_list
    print(f"\nLists match: {lists_match}")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
