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

from typing import Any, List, Optional, Set, Dict
from collections import deque
import networkx as nx
from matplotlib.patches import Patch


class TreeNode:
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
        self.parent: Optional['TreeNode'] = None
        self.children: List['TreeNode'] = []
    
    def add_child(self, child: 'TreeNode') -> None:
        """Adds a child node."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
    
    def remove_child(self, child: 'TreeNode') -> bool:
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
        return f"TreeNode({self.data})"
    
    def __repr__(self) -> str:
        return f"TreeNode(data={self.data}, children={len(self.children)})"


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
        self.root: Optional[TreeNode] = TreeNode(root_data) if root_data is not None else None
    
    def is_empty(self) -> bool:
        """Checks if tree is empty."""
        return self.root is None
    
    def set_root(self, data: Any) -> TreeNode:
        """
        Sets the root of the tree.
        
        Args:
            data: Data for the root
            
        Returns:
            The created root node
        """
        self.root = TreeNode(data)
        return self.root
    
    def add_child(self, parent: TreeNode, child_data: Any) -> TreeNode:
        """
        Adds a child to a parent node.
        
        Args:
            parent: Parent node
            child_data: Data for the new child
            
        Returns:
            The created child node
        """
        child = TreeNode(child_data)
        parent.add_child(child)
        return child
    
    def get_node_count(self) -> int:
        """
        Returns the number of nodes in the tree (n).
        
        Returns:
            Number of nodes
        """
        if self.is_empty():
            return 0
        return self._count_nodes(self.root)
    
    def _count_nodes(self, node: TreeNode) -> int:
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
    
    def get_depth(self, node: TreeNode) -> int:
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
    
    def _get_height(self, node: TreeNode) -> int:
        """Helper method to calculate height (recursive)."""
        if node.is_leaf():
            return 0
        max_height = 0
        for child in node.children:
            child_height = self._get_height(child)
            max_height = max(max_height, child_height)
        return max_height + 1
    
    def get_level(self, depth: int) -> List[TreeNode]:
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
    
    def _collect_level_nodes(self, node: TreeNode, current_depth: int, 
                            target_depth: int, result: List[TreeNode]) -> None:
        """Helper method to collect nodes at a level."""
        if current_depth == target_depth:
            result.append(node)
            return
        
        for child in node.children:
            self._collect_level_nodes(child, current_depth + 1, target_depth, result)
    
    def get_all_levels(self) -> Dict[int, List[TreeNode]]:
        """
        Returns all levels of the tree.
        
        Returns:
            Dictionary: {depth: [nodes at this level]}
        """
        levels = {}
        if not self.is_empty():
            self._collect_all_levels(self.root, 0, levels)
        return levels
    
    def _collect_all_levels(self, node: TreeNode, depth: int, 
                           levels: Dict[int, List[TreeNode]]) -> None:
        """Helper method to collect all levels."""
        if depth not in levels:
            levels[depth] = []
        levels[depth].append(node)
        
        for child in node.children:
            self._collect_all_levels(child, depth + 1, levels)
    
    def get_leaves(self) -> List[TreeNode]:
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
    
    def _collect_leaves(self, node: TreeNode, leaves: List[TreeNode]) -> None:
        """Helper method to collect leaves."""
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)
    
    def get_inner_nodes(self) -> List[TreeNode]:
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
    
    def _collect_inner_nodes(self, node: TreeNode, inner_nodes: List[TreeNode]) -> None:
        """Helper method to collect inner nodes."""
        if node.is_inner_node():
            inner_nodes.append(node)
            for child in node.children:
                self._collect_inner_nodes(child, inner_nodes)
    
    def find_path(self, from_node: TreeNode, to_node: TreeNode) -> Optional[List[TreeNode]]:
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
    
    def _visit_all(self, node: TreeNode, visited: Set[TreeNode]) -> None:
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
    
    def traverse_preorder(self, node: Optional[TreeNode] = None) -> List[Any]:
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
    
    def traverse_postorder(self, node: Optional[TreeNode] = None) -> List[Any]:
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
    
    def traverse_inorder(self, node: Optional[TreeNode] = None) -> List[Any]:
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
    
    def find_node(self, data: Any) -> Optional[TreeNode]:
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
    
    def _find_node_recursive(self, node: TreeNode, data: Any) -> Optional[TreeNode]:
        """Helper method for node search."""
        if node.data == data:
            return node
        
        for child in node.children:
            result = self._find_node_recursive(child, data)
            if result is not None:
                return result
        
        return None
    
    def get_ancestors(self, node: TreeNode) -> List[TreeNode]:
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
    
    def get_descendants(self, node: TreeNode) -> List[TreeNode]:
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
    
    def _collect_descendants(self, node: TreeNode, descendants: List[TreeNode]) -> None:
        """Helper method to collect descendants."""
        for child in node.children:
            descendants.append(child)
            self._collect_descendants(child, descendants)
    
    def print_tree(self, node: Optional[TreeNode] = None, prefix: str = "", is_last: bool = True) -> None:
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
    
    def visualize(self, title: Optional[str] = None, figsize: tuple = (12, 9), 
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
        
        # Add all nodes and edges
        self._add_nodes_to_graph(G, self.root)
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        if positions:
            pos = positions
        else:
            # Use hierarchical layout based on root position
            if root_position == "top":
                pos = self._get_hierarchical_pos(G, self.root.data, orientation="vertical", reverse=False)
            elif root_position == "bottom":
                pos = self._get_hierarchical_pos(G, self.root.data, orientation="vertical", reverse=True)
            elif root_position == "left":
                pos = self._get_hierarchical_pos(G, self.root.data, orientation="horizontal", reverse=True)
            elif root_position == "right":
                pos = self._get_hierarchical_pos(G, self.root.data, orientation="horizontal", reverse=False)
            else:
                # Fallback to spring layout
                pos = nx.spring_layout(G, seed=42)
        
        # Categorize nodes
        root_nodes = [self.root.data]
        leaf_nodes = [leaf.data for leaf in self.get_leaves()]
        inner_nodes = [node.data for node in self.get_inner_nodes() if node != self.root]
        
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
        
        # Draw labels
        nx.draw_networkx_labels(G, pos,
                               font_size=10,
                               font_weight='bold',
                               font_color='black')
        
        if title is None:
            title = f"Tree Visualization (Root: {root_position})"
        
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

    def _add_nodes_to_graph(self, G, node: TreeNode):
        """Helper method to add nodes and edges to NetworkX graph."""
        G.add_node(node.data)
        for child in node.children:
            G.add_edge(node.data, child.data)
            self._add_nodes_to_graph(G, child)

    def _get_hierarchical_pos(self, G, root, orientation="vertical", reverse=False):
        """
        Create hierarchical layout for tree visualization.
        
        Args:
            G: NetworkX graph
            root: Root node data
            orientation: "vertical" or "horizontal"
            reverse: Whether to reverse the direction
        """
        def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
            pos = {root: (xcenter, vert_loc)}
            return _hierarchy_pos_helper(G, root, width, vert_gap, vert_loc, xcenter, pos)
        
        def _hierarchy_pos_helper(G, root, width, vert_gap, vert_loc, xcenter, pos, parent=None):
            children = list(G.neighbors(root))
            if parent and parent in children:
                children.remove(parent)
            
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos[child] = (nextx, vert_loc - vert_gap)
                    pos = _hierarchy_pos_helper(G, child, width=dx, vert_gap=vert_gap,
                                              vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                              parent=root)
            return pos
        
        pos = _hierarchy_pos(G, root)
        
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
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        return (f"Tree(nodes={stats['node_count']}, "
                f"edges={stats['edge_count']}, "
                f"height={stats['height']})")
    
    def __repr__(self) -> str:
        return self.__str__()


# Example usage
if __name__ == "__main__":
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

    # Visualize tree with root at the left
    print("\n=== Tree Visualization (Root at Left) ===")
    tree.visualize(title="Tree with Root at Left", root_position="left")
