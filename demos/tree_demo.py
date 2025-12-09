"""
Tree Data Structure - Comprehensive Demo

This demo showcases the complete functionality of the Tree class,
including creation, manipulation, traversal, and various operations.

Topics covered:
1. Manual tree creation
2. Import from adjacency matrix
3. Import from adjacency list  
4. Import from nested structure
5. Tree traversals (preorder, inorder, postorder, level-order)
6. Generator-based iteration (memory-efficient)
7. Tree properties and statistics
8. Search and path finding
9. Export operations
10. Round-trip conversions
11. Visualization

Author: PyHelper JKluess
Date: December 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhelper_jkluess.Complex.Trees.tree import Tree


def example_1_manual_creation():
    """
    Example 1: Creating a tree manually
    
    Demonstrates:
    - Creating a tree with a root
    - Adding children to nodes
    - Building a multi-level tree structure
    """
    print("=" * 80)
    print("EXAMPLE 1: Manual Tree Creation")
    print("=" * 80)
    
    # Create tree with root
    tree = Tree("Root")
    print("✓ Created tree with root 'Root'")
    
    # Add children to root
    child_a = tree.add_child(tree.root, "A")
    child_b = tree.add_child(tree.root, "B")
    child_c = tree.add_child(tree.root, "C")
    print("✓ Added children A, B, C to root")
    
    # Add grandchildren to A
    child_a1 = tree.add_child(child_a, "A1")
    child_a2 = tree.add_child(child_a, "A2")
    print("✓ Added children A1, A2 to node A")
    
    # Add child to B
    child_b1 = tree.add_child(child_b, "B1")
    print("✓ Added child B1 to node B")
    
    # Visualize the tree structure
    print("\nTree Structure:")
    tree.print_tree()
    
    return tree


def example_2_tree_properties(tree):
    """
    Example 2: Exploring tree properties
    
    Demonstrates:
    - Node counting
    - Edge counting
    - Tree property verification (m = n - 1)
    - Height calculation
    - Depth calculation
    - Degree calculation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Tree Properties")
    print("=" * 80)
    
    print(f"\n📊 Basic Metrics:")
    print(f"  Nodes (n): {tree.get_node_count()}")
    print(f"  Edges (m): {tree.get_edge_count()}")
    print(f"  Height: {tree.get_height()}")
    
    print(f"\n✓ Tree Property (m = n - 1): {tree.verify_tree_property()}")
    
    # Node-specific properties
    node_a = tree.find_node("A")
    if node_a:
        print(f"\n🔍 Properties of Node 'A':")
        print(f"  Depth: {tree.get_depth(node_a)}")
        print(f"  Degree: {tree.get_degree(node_a)}")
        print(f"  Is Leaf: {node_a.is_leaf()}")
        print(f"  Is Inner Node: {node_a.is_inner_node()}")


def example_3_node_classification(tree):
    """
    Example 3: Node classification
    
    Demonstrates:
    - Finding all leaves
    - Finding all inner nodes
    - Getting nodes by level
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Node Classification")
    print("=" * 80)
    
    # Get leaves
    leaves = tree.get_leaves()
    print(f"\n🍃 Leaves: {[leaf.data for leaf in leaves]}")
    
    # Get inner nodes
    inner = tree.get_inner_nodes()
    print(f"🔸 Inner Nodes: {[node.data for node in inner]}")
    
    # Get nodes by level
    print(f"\n📊 Nodes by Level:")
    levels = tree.get_all_levels()
    for depth, nodes in levels.items():
        print(f"  Level {depth}: {[node.data for node in nodes]}")


def example_4_traversals(tree):
    """
    Example 4: Tree traversals
    
    Demonstrates:
    - Preorder traversal (Root → Children)
    - Inorder traversal (First child → Root → Others)
    - Postorder traversal (Children → Root)
    - Level-order traversal (Breadth-first)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Tree Traversals")
    print("=" * 80)
    
    print("\n🔄 List-based Traversals (return complete list):")
    print(f"  Preorder (Root→Children):     {tree.traverse_preorder()}")
    print(f"  Inorder (First→Root→Others):  {tree.traverse_inorder()}")
    print(f"  Postorder (Children→Root):    {tree.traverse_postorder()}")
    print(f"  Level-order (BFS):            {tree.traverse_levelorder()}")


def example_5_generators(tree):
    """
    Example 5: Generator-based iteration
    
    Demonstrates:
    - Memory-efficient iteration
    - Early stopping capability
    - Using generators for large trees
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Generator-based Iteration")
    print("=" * 80)
    
    print("\n🔄 Generator Traversals (yield one at a time):")
    
    # Preorder generator
    print("\n  Preorder with early stopping:")
    for i, value in enumerate(tree.iter_preorder()):
        print(f"    {i+1}. {value}")
        if i >= 3:  # Stop after 4 nodes
            print("    ... (stopped early)")
            break
    
    # Level-order generator
    print("\n  Level-order generator:")
    level_order_gen = tree.iter_levelorder()
    print(f"    First 3 nodes: {[next(level_order_gen) for _ in range(3)]}")
    
    # Use case: finding first match
    print("\n  Finding first node containing '1':")
    for value in tree.iter_preorder():
        if '1' in str(value):
            print(f"    Found: {value}")
            break


def example_6_search_and_paths(tree):
    """
    Example 6: Search and path operations
    
    Demonstrates:
    - Finding nodes by value
    - Getting ancestors
    - Getting descendants
    - Finding paths between nodes
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Search and Path Operations")
    print("=" * 80)
    
    # Find node
    target = tree.find_node("A2")
    print(f"\n🔍 Found node: {target.data if target else 'Not found'}")
    
    if target:
        # Get ancestors
        ancestors = tree.get_ancestors(target)
        print(f"  Ancestors of A2: {[node.data for node in ancestors]}")
        
        # Get descendants
        node_a = tree.find_node("A")
        if node_a:
            descendants = tree.get_descendants(node_a)
            print(f"  Descendants of A: {[node.data for node in descendants]}")
    
    # Find path between nodes
    node_a2 = tree.find_node("A2")
    node_b1 = tree.find_node("B1")
    if node_a2 and node_b1:
        path = tree.find_path(node_a2, node_b1)
        if path:
            print(f"\n🛤️  Path from A2 to B1: {[node.data for node in path]}")


def example_7_statistics(tree):
    """
    Example 7: Complete statistics
    
    Demonstrates:
    - Getting all statistics at once
    - Understanding tree metrics
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Complete Statistics")
    print("=" * 80)
    
    stats = tree.get_statistics()
    
    print("\n📊 Complete Tree Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_8_adjacency_matrix():
    """
    Example 8: Creating tree from adjacency matrix
    
    Demonstrates:
    - Matrix representation of trees
    - Creating tree from matrix
    - Understanding matrix format
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Adjacency Matrix Import")
    print("=" * 80)
    
    # Define adjacency matrix
    # matrix[i][j] = 1 means node i is parent of node j
    matrix = [
        [0, 1, 1, 0, 0],  # Node 0 (Root) → nodes 1, 2
        [0, 0, 0, 1, 1],  # Node 1 (A) → nodes 3, 4
        [0, 0, 0, 0, 0],  # Node 2 (B) is leaf
        [0, 0, 0, 0, 0],  # Node 3 (A1) is leaf
        [0, 0, 0, 0, 0]   # Node 4 (A2) is leaf
    ]
    
    labels = ['Root', 'A', 'B', 'A1', 'A2']
    
    print("\n📋 Adjacency Matrix:")
    print("   ", "  ".join(labels))
    for i, row in enumerate(matrix):
        print(f"{labels[i]:5s} {row}")
    
    # Create tree from matrix
    tree = Tree.from_adjacency_matrix(matrix, labels)
    
    print("\n🌳 Resulting Tree:")
    tree.print_tree()
    
    print(f"\n✓ Nodes: {tree.get_node_count()}, Edges: {tree.get_edge_count()}")
    
    return tree


def example_9_adjacency_list():
    """
    Example 9: Creating tree from adjacency list
    
    Demonstrates:
    - Dictionary representation of trees
    - Creating tree from adjacency list
    - More intuitive format for manual creation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Adjacency List Import")
    print("=" * 80)
    
    # Define adjacency list
    adj_list = {
        'Root': ['A', 'B', 'C'],
        'A': ['A1', 'A2'],
        'B': ['B1'],
        'C': [],          # Leaf node
        'A1': [],
        'A2': [],
        'B1': []
    }
    
    print("\n📋 Adjacency List:")
    for parent, children in adj_list.items():
        children_str = ', '.join(children) if children else 'leaf'
        print(f"  {parent} → [{children_str}]")
    
    # Create tree from adjacency list
    tree = Tree.from_adjacency_list(adj_list, 'Root')
    
    print("\n🌳 Resulting Tree:")
    tree.print_tree()
    
    return tree


def example_10_nested_structure():
    """
    Example 10: Creating tree from nested structure
    
    Demonstrates:
    - Tuple/list representation
    - Most intuitive format
    - Handling nested hierarchies
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Nested Structure Import")
    print("=" * 80)
    
    # Define nested structure
    # Format: (root_value, [child1, child2, ...])
    structure = ('Root', [
        ('A', [('A1', []), ('A2', [])]),
        ('B', [('B1', [])]),
        ('C', [])
    ])
    
    print(f"\n📋 Nested Structure:")
    print(f"  {structure}")
    
    # Create tree from nested structure
    tree = Tree.from_nested_structure(structure)
    
    print("\n🌳 Resulting Tree:")
    tree.print_tree()
    
    # Show traversal
    print(f"\n🔄 Preorder: {tree.traverse_preorder()}")
    
    return tree


def example_11_export_operations(tree):
    """
    Example 11: Exporting trees
    
    Demonstrates:
    - Exporting to adjacency matrix
    - Exporting to adjacency list
    - Exporting to nested structure
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 11: Export Operations")
    print("=" * 80)
    
    print("\n🌳 Original Tree:")
    tree.print_tree()
    
    # Export to adjacency matrix
    print("\n📋 Export to Adjacency Matrix:")
    matrix = tree.get_adjacency_matrix()
    labels = tree.get_node_labels()
    print("   ", "  ".join(str(l) for l in labels))
    for i, row in enumerate(matrix):
        print(f"{str(labels[i]):5s} {row}")
    
    # Export to adjacency list
    print("\n📋 Export to Adjacency List:")
    adj_list = tree.get_adjacency_list()
    for parent, children in adj_list.items():
        print(f"  {parent} → {children}")
    
    # Export to nested structure
    print("\n📋 Export to Nested Structure:")
    nested = tree.to_nested_structure()
    print(f"  {nested}")


def example_12_round_trip():
    """
    Example 12: Round-trip conversion
    
    Demonstrates:
    - Converting between formats
    - Verifying data integrity
    - Understanding format differences
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 12: Round-trip Conversion")
    print("=" * 80)
    
    # Original adjacency list
    original_list = {
        'P': ['Q', 'R'],
        'Q': ['S'],
        'R': [],
        'S': []
    }
    
    print("\n📋 Original Adjacency List:")
    for k, v in original_list.items():
        print(f"  {k} → {v}")
    
    # Create tree
    tree = Tree.from_adjacency_list(original_list, 'P')
    print("\n🌳 Tree:")
    tree.print_tree()
    
    # Export back
    exported_list = tree.get_adjacency_list()
    print("\n📋 Exported Adjacency List:")
    for k, v in exported_list.items():
        print(f"  {k} → {v}")
    
    # Verify
    match = original_list == exported_list
    print(f"\n✓ Lists match: {match}")


def example_13_graph_properties(tree):
    """
    Example 13: Graph theory properties
    
    Demonstrates:
    - Connectivity checking
    - Cycle detection
    - Acyclic verification
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 13: Graph Theory Properties")
    print("=" * 80)
    
    print(f"\n📊 Graph Properties:")
    print(f"  Is Connected: {tree.is_connected()}")
    print(f"  Has Cycle: {tree.has_cycle()}")
    print(f"  Is Acyclic: {tree.is_acyclic()}")
    
    print(f"\n💡 Trees are always:")
    print(f"  • Connected (single component)")
    print(f"  • Acyclic (no cycles)")
    print(f"  • Have exactly n-1 edges for n nodes")


def example_14_visualization(tree):
    """
    Example 14: Tree visualization
    
    Demonstrates:
    - ASCII visualization
    - Matplotlib graph visualization
    - Different root positions
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 14: Tree Visualization")
    print("=" * 80)
    
    print("\n📊 ASCII Visualization:")
    tree.print_tree()
    
    print("\n📊 Matplotlib Visualization:")
    print("  • visualize(root_position='top')    - Root at top (default)")
    print("  • visualize(root_position='bottom') - Root at bottom")
    print("  • visualize(root_position='left')   - Root at left")
    print("  • visualize(root_position='right')  - Root at right")
    
    # Uncomment to show visualization:
    # tree.visualize(title="Tree Visualization", root_position="top")


def main():
    """Run all examples in sequence"""
    import sys
    import io
    # Fix Windows console encoding for emoji support
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n")
    print("⭐" * 40)
    print("TREE DATA STRUCTURE - COMPREHENSIVE DEMO")
    print("🌳" * 40)
    
    # Run all examples
    tree1 = example_1_manual_creation()
    example_2_tree_properties(tree1)
    example_3_node_classification(tree1)
    example_4_traversals(tree1)
    example_5_generators(tree1)
    example_6_search_and_paths(tree1)
    example_7_statistics(tree1)
    
    tree2 = example_8_adjacency_matrix()
    tree3 = example_9_adjacency_list()
    tree4 = example_10_nested_structure()
    
    example_11_export_operations(tree1)
    example_12_round_trip()
    example_13_graph_properties(tree1)
    example_14_visualization(tree1)
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)
    print("\n📚 Key Takeaways:")
    print("  1. Trees are connected, acyclic graphs with n-1 edges")
    print("  2. Multiple ways to create trees (manual, matrix, list, nested)")
    print("  3. Different traversals serve different purposes")
    print("  4. Generators are memory-efficient for large trees")
    print("  5. Trees support rich graph theory operations")
    print("\n")


if __name__ == "__main__":
    main()
