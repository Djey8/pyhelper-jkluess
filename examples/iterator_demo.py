"""
Demo: Using iterators for Trees and Graphs

This demonstrates how to iterate through nodes in different orders
using the existing traversal methods and new generator-based iterators.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhelper_jkluess.Complex.Trees import Tree, BinaryTree
from pyhelper_jkluess.Complex.Graphs import UndirectedGraph


def demo_tree_iteration():
    """Demonstrate tree iteration methods"""
    print("=" * 80)
    print("TREE ITERATION DEMO")
    print("=" * 80)
    
    # Create a sample tree
    tree = Tree("Root")
    a = tree.add_child(tree.root, "A")
    b = tree.add_child(tree.root, "B")
    c = tree.add_child(tree.root, "C")
    tree.add_child(a, "A1")
    tree.add_child(a, "A2")
    tree.add_child(b, "B1")
    
    print("\nTree structure:")
    tree.print_tree()
    
    # Method 1: Using existing traverse methods (returns list)
    print("\n1. Preorder traversal (Root -> Children):")
    print("   ", tree.traverse_preorder())
    
    print("\n2. Postorder traversal (Children -> Root):")
    print("   ", tree.traverse_postorder())
    
    print("\n3. Level-order traversal (Breadth-first):")
    print("   ", tree.traverse_levelorder())
    
    print("\n4. Inorder traversal:")
    print("   ", tree.traverse_inorder())
    
    # Method 2: Using for loops (more Pythonic)
    print("\n5. Iterate with for loop (preorder):")
    for data in tree.traverse_preorder():
        print(f"   Processing node: {data}")


def demo_binary_tree_iteration():
    """Demonstrate binary tree iteration methods"""
    print("\n" + "=" * 80)
    print("BINARY TREE ITERATION DEMO")
    print("=" * 80)
    
    # Create a binary tree
    tree = BinaryTree(10)
    tree.insert_left(tree.root, 5)
    tree.insert_right(tree.root, 15)
    tree.insert_left(tree.root.left, 3)
    tree.insert_right(tree.root.left, 7)
    tree.insert_left(tree.root.right, 12)
    tree.insert_right(tree.root.right, 20)
    
    print("\nBinary Tree structure:")
    tree.print_tree()
    
    print("\n1. Preorder traversal (Root -> Left -> Right):")
    print("   ", tree.traverse_preorder())
    
    print("\n2. Inorder traversal (Left -> Root -> Right):")
    print("   ", tree.traverse_inorder())
    
    print("\n3. Postorder traversal (Left -> Right -> Root):")
    print("   ", tree.traverse_postorder())
    
    print("\n4. Level-order traversal (Breadth-first):")
    print("   ", tree.traverse_levelorder())


def demo_graph_iteration():
    """Demonstrate graph iteration methods"""
    print("\n" + "=" * 80)
    print("GRAPH ITERATION DEMO")
    print("=" * 80)
    
    # Create a sample graph
    graph = UndirectedGraph()
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "E")
    graph.add_edge("C", "F")
    
    print("\nGraph structure:")
    print(graph)
    
    print("\n1. DFS (Depth-First Search) from 'A':")
    print("   ", graph.dfs("A"))
    
    print("\n2. BFS (Breadth-First Search) from 'A':")
    print("   ", graph.bfs("A"))
    
    print("\n3. Iterate through all vertices:")
    for vertex in graph.get_vertices():
        neighbors = graph.get_neighbors(vertex)
        print(f"   {vertex}: neighbors = {neighbors}")


def demo_custom_generators():
    """Demonstrate custom generator-based iterators"""
    print("\n" + "=" * 80)
    print("CUSTOM GENERATOR ITERATORS DEMO")
    print("=" * 80)
    
    tree = BinaryTree(1)
    tree.insert_left(tree.root, 2)
    tree.insert_right(tree.root, 3)
    tree.insert_left(tree.root.left, 4)
    tree.insert_right(tree.root.left, 5)
    
    print("\nBinary Tree:")
    tree.print_tree()
    
    # Generator function for preorder
    def preorder_generator(node):
        """Generator for preorder traversal"""
        if node is not None:
            yield node.data
            if hasattr(node, 'left'):
                yield from preorder_generator(node.left)
            if hasattr(node, 'right'):
                yield from preorder_generator(node.right)
    
    # Generator function for inorder
    def inorder_generator(node):
        """Generator for inorder traversal"""
        if node is not None:
            if hasattr(node, 'left'):
                yield from inorder_generator(node.left)
            yield node.data
            if hasattr(node, 'right'):
                yield from inorder_generator(node.right)
    
    # Generator function for postorder
    def postorder_generator(node):
        """Generator for postorder traversal"""
        if node is not None:
            if hasattr(node, 'left'):
                yield from postorder_generator(node.left)
            if hasattr(node, 'right'):
                yield from postorder_generator(node.right)
            yield node.data
    
    # Generator function for level-order
    def levelorder_generator(root):
        """Generator for level-order traversal"""
        if root is None:
            return
        
        from collections import deque
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            yield node.data
            
            if hasattr(node, 'left') and node.left:
                queue.append(node.left)
            if hasattr(node, 'right') and node.right:
                queue.append(node.right)
    
    print("\n1. Using preorder generator:")
    for value in preorder_generator(tree.root):
        print(f"   Visited: {value}")
    
    print("\n2. Using inorder generator:")
    for value in inorder_generator(tree.root):
        print(f"   Visited: {value}")
    
    print("\n3. Using postorder generator:")
    for value in postorder_generator(tree.root):
        print(f"   Visited: {value}")
    
    print("\n4. Using level-order generator:")
    for value in levelorder_generator(tree.root):
        print(f"   Visited: {value}")
    
    # Generator advantage: Can stop early
    print("\n5. Generator advantage - stop early (first 3 nodes):")
    count = 0
    for value in preorder_generator(tree.root):
        print(f"   Visited: {value}")
        count += 1
        if count >= 3:
            print("   Stopping early!")
            break


def demo_leaves_only():
    """Demonstrate iterating through leaves only"""
    print("\n" + "=" * 80)
    print("LEAVES ITERATION DEMO")
    print("=" * 80)
    
    tree = Tree("Root")
    a = tree.add_child(tree.root, "A")
    b = tree.add_child(tree.root, "B")
    c = tree.add_child(tree.root, "C")
    tree.add_child(a, "A1")
    tree.add_child(a, "A2")
    tree.add_child(b, "B1")
    
    print("\nTree structure:")
    tree.print_tree()
    
    print("\n1. All leaves (built-in method):")
    leaves = tree.get_leaves()
    print("   ", [leaf.data for leaf in leaves])
    
    print("\n2. Iterate through leaves only:")
    for leaf in leaves:
        print(f"   Leaf: {leaf.data}, Parent: {leaf.parent.data if leaf.parent else 'None'}")
    
    print("\n3. All inner nodes:")
    inner = tree.get_inner_nodes()
    print("   ", [node.data for node in inner])


if __name__ == "__main__":
    demo_tree_iteration()
    demo_binary_tree_iteration()
    demo_graph_iteration()
    demo_custom_generators()
    demo_leaves_only()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Available iteration methods:

TREES:
- tree.traverse_preorder()    : Root -> Children (DFS)
- tree.traverse_postorder()   : Children -> Root (DFS)
- tree.traverse_levelorder()  : Level-by-level (BFS)
- tree.traverse_inorder()     : Left -> Root -> Right
- tree.get_leaves()           : Only leaf nodes
- tree.get_inner_nodes()      : Only inner nodes

BINARY TREES:
- tree.traverse_preorder()    : Root -> Left -> Right
- tree.traverse_inorder()     : Left -> Root -> Right (sorted for BST)
- tree.traverse_postorder()   : Left -> Right -> Root
- tree.traverse_levelorder()  : Level-by-level (BFS)

GRAPHS:
- graph.dfs(start)            : Depth-first search
- graph.bfs(start)            : Breadth-first search
- graph.get_vertices()        : All vertices
- graph.get_neighbors(v)      : Neighbors of vertex v

For custom iteration orders, you can create generator functions
that yield nodes one at a time (see demo_custom_generators).
    """)
