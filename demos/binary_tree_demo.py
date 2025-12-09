"""
Binary Tree Data Structure - Comprehensive Demo

This demo showcases the complete functionality of the BinaryTree class,
including creation, properties, traversals, and specialized operations.

Topics covered:
1. Manual binary tree creation
2. Binary tree from nested structure (RECOMMENDED)
3. Expression trees with operators
4. Binary tree properties (complete, perfect, balanced)
5. Binary-specific traversals
6. Import/export operations
7. Binary Search Tree (BST) concepts
8. Tree sorting algorithm

Author: PyHelper JKluess
Date: December 2025
"""

from pyhelper_jkluess.Complex.Trees.binary_tree import BinaryTree


def example_1_manual_creation():
    """
    Example 1: Creating a binary tree manually
    
    Demonstrates:
    - Creating binary tree with root
    - Using insert_left and insert_right
    - Building multi-level structure
    """
    print("=" * 80)
    print("EXAMPLE 1: Manual Binary Tree Creation")
    print("=" * 80)
    
    # Create binary tree with root
    tree = BinaryTree(1)
    print("✓ Created binary tree with root value 1")
    
    # Add left and right children to root
    left_child = tree.insert_left(tree.root, 2)
    right_child = tree.insert_right(tree.root, 3)
    print("✓ Added left child (2) and right child (3) to root")
    
    # Add children to left subtree
    tree.insert_left(left_child, 4)
    tree.insert_right(left_child, 5)
    print("✓ Added children (4, 5) to left subtree")
    
    # Add children to right subtree
    tree.insert_left(right_child, 6)
    tree.insert_right(right_child, 7)
    print("✓ Added children (6, 7) to right subtree")
    
    print("\n🌳 Binary Tree Structure:")
    tree.print_tree()
    
    print(f"\n📊 Tree Properties:")
    print(f"  Nodes: {tree.get_node_count()}")
    print(f"  Height: {tree.get_height()}")
    print(f"  Leaf Count: {tree.get_leaf_count()}")
    
    return tree


def example_2_binary_traversals(tree):
    """
    Example 2: Binary tree traversals
    
    Demonstrates:
    - Preorder (Root → Left → Right)
    - Inorder (Left → Root → Right)
    - Postorder (Left → Right → Root)
    - Level-order (Breadth-first)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Binary Tree Traversals")
    print("=" * 80)
    
    print("\n🔄 Tree Structure:")
    tree.print_tree()
    
    print("\n📋 Traversal Results:")
    print(f"  Preorder  (Root→Left→Right):  {tree.traverse_preorder()}")
    print(f"  Inorder   (Left→Root→Right):  {tree.traverse_inorder()}")
    print(f"  Postorder (Left→Right→Root):  {tree.traverse_postorder()}")
    print(f"  Levelorder (Breadth-first):   {tree.traverse_levelorder()}")
    
    print("\n💡 Use Cases:")
    print("  • Preorder:  Copying tree, prefix expression")
    print("  • Inorder:   Sorted output for BST, infix expression")
    print("  • Postorder: Deleting tree, postfix expression")
    print("  • Levelorder: BFS, shortest path, serialization")


def example_3_generators(tree):
    """
    Example 3: Generator-based iteration (inherited from Tree)
    
    Demonstrates:
    - Memory-efficient iteration
    - Early stopping
    - Generator advantages
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Generator-based Iteration")
    print("=" * 80)
    
    print("\n🔄 Generators yield one node at a time:")
    
    # Inorder generator with early stopping
    print("\n  Inorder (first 4 nodes):")
    for i, value in enumerate(tree.iter_inorder()):
        print(f"    {i+1}. {value}")
        if i >= 3:
            print("    ... (stopped early)")
            break
    
    # Finding first even number
    print("\n  Finding first even number (preorder):")
    for value in tree.iter_preorder():
        if value % 2 == 0:
            print(f"    Found: {value}")
            break
    
    print("\n💡 Generators are memory-efficient for large trees")


def example_4_binary_properties():
    """
    Example 4: Binary tree properties
    
    Demonstrates:
    - Complete binary tree
    - Perfect binary tree
    - Balanced binary tree
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Binary Tree Properties")
    print("=" * 80)
    
    # Complete binary tree
    print("\n📐 Complete Binary Tree:")
    print("  Definition: Every level fully filled except possibly last")
    complete_tree = BinaryTree.from_nested_structure((1, [2, 3]))
    complete_tree.print_tree()
    print(f"  Is Complete: {complete_tree.is_complete()}")
    
    # Perfect binary tree
    print("\n📐 Perfect Binary Tree:")
    print("  Definition: All internal nodes have 2 children, all leaves at same level")
    perfect_tree = BinaryTree.from_nested_structure((1, [(2, [4, 5]), (3, [6, 7])]))
    perfect_tree.print_tree()
    print(f"  Is Perfect: {perfect_tree.is_perfect()}")
    print(f"  Is Complete: {perfect_tree.is_complete()}")
    print(f"  Nodes: {perfect_tree.get_node_count()} = 2^{perfect_tree.get_height()+1} - 1")
    
    # Balanced binary tree
    print("\n📐 Balanced Binary Tree:")
    print("  Definition: Height difference between any two leaves ≤ 1")
    balanced_tree = BinaryTree.from_nested_structure((1, [(2, [4, 5]), 3]))
    balanced_tree.print_tree()
    print(f"  Is Balanced: {balanced_tree.is_balanced()}")
    
    # Unbalanced tree
    print("\n📐 Unbalanced Binary Tree:")
    unbalanced_tree = BinaryTree.from_nested_structure((1, [(2, [(3, [4, None]), None]), None]))
    unbalanced_tree.print_tree()
    print(f"  Is Balanced: {unbalanced_tree.is_balanced()}")


def example_5_nested_structure():
    """
    Example 5: Creating trees from nested structure (RECOMMENDED)
    
    Demonstrates:
    - Intuitive tuple/list format
    - Easy tree construction
    - Handling various structures
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Nested Structure Format (RECOMMENDED)")
    print("=" * 80)
    
    print("\n📋 Format: (root, [left_subtree, right_subtree])")
    
    # Simple tree
    print("\n🌳 Simple Tree:")
    structure1 = (1, [2, 3])
    print(f"  Structure: {structure1}")
    tree1 = BinaryTree.from_nested_structure(structure1)
    tree1.print_tree()
    
    # Complex tree
    print("\n🌳 Complex Tree:")
    structure2 = (1, [(2, [4, 5]), (3, [6, 7])])
    print(f"  Structure: {structure2}")
    tree2 = BinaryTree.from_nested_structure(structure2)
    tree2.print_tree()
    
    # Asymmetric tree (None for missing children)
    print("\n🌳 Asymmetric Tree:")
    structure3 = (1, [(2, [4, None]), None])
    print(f"  Structure: {structure3}")
    tree3 = BinaryTree.from_nested_structure(structure3)
    tree3.print_tree()
    
    return tree2


def example_6_expression_trees():
    """
    Example 6: Expression trees
    
    Demonstrates:
    - Representing mathematical expressions
    - Different notations (prefix, infix, postfix)
    - Evaluating expressions
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Expression Trees")
    print("=" * 80)
    
    # Expression: (4 + 5) * (2 - 1)
    print("\n🧮 Expression: (4 + 5) * (2 - 1)")
    structure = ('*', [('+', [4, 5]), ('-', [2, 1])])
    print(f"  Structure: {structure}")
    
    expr_tree = BinaryTree.from_nested_structure(structure)
    expr_tree.print_tree()
    
    print("\n📋 Different Notations:")
    print(f"  Prefix (Preorder):   {expr_tree.traverse_preorder()}")
    print(f"  Infix (Inorder):     {expr_tree.traverse_inorder()}")
    print(f"  Postfix (Postorder): {expr_tree.traverse_postorder()}")
    
    print("\n💡 Evaluation Order:")
    print("  Prefix:  * + 4 5 - 2 1  (operator first)")
    print("  Infix:   4 + 5 * 2 - 1  (operator between operands)")
    print("  Postfix: 4 5 + 2 1 - *  (operator last)")
    
    # Another expression: (3 + 5) * 3
    print("\n🧮 Expression with Duplicate Values: (3 + 5) * 3")
    structure2 = ('*', [('+', [3, 5]), 3])
    expr_tree2 = BinaryTree.from_nested_structure(structure2)
    expr_tree2.print_tree()
    print(f"  Postfix: {expr_tree2.traverse_postorder()}")
    print("  Note: Two different nodes with value 3")


def example_7_bst_sorting():
    """
    Example 7: Binary Search Tree (BST) and Tree Sorting
    
    Demonstrates:
    - BST property
    - Tree sorting algorithm
    - Inorder gives sorted output
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Binary Search Tree (BST) Sorting")
    print("=" * 80)
    
    print("\n📋 Tree Sorting Algorithm:")
    print("  1. Insert values comparing with parent (left if <, right if >)")
    print("  2. Ignore duplicates")
    print("  3. Inorder traversal gives sorted output")
    
    # Manually build BST
    print("\n🌳 Building BST with values: [5, 3, 7, 1, 9, 2, 8]")
    
    bst = BinaryTree(5)
    
    # Insert values following BST property
    # This would normally use a BST insert method
    # Here we manually construct for demonstration
    bst.insert_left(bst.root, 3)
    bst.insert_right(bst.root, 7)
    
    left = bst.root.left
    bst.insert_left(left, 1)
    
    left_left = left.left
    bst.insert_right(left_left, 2)
    
    right = bst.root.right
    bst.insert_right(right, 9)
    bst.insert_left(right, 8)  # Note: Not following BST property perfectly for demo
    
    bst.print_tree()
    
    print(f"\n📋 Inorder Traversal (should be sorted for perfect BST):")
    print(f"  {bst.traverse_inorder()}")
    
    print("\n💡 For proper BST:")
    print("  • Left subtree values < Root value")
    print("  • Right subtree values > Root value")
    print("  • Inorder traversal = Sorted order")


def example_8_import_export():
    """
    Example 8: Import/Export operations
    
    Demonstrates:
    - Creating from adjacency matrix
    - Creating from adjacency list
    - Exporting to different formats
    - Round-trip conversion
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Import/Export Operations")
    print("=" * 80)
    
    # Create from adjacency matrix
    print("\n📋 From Adjacency Matrix:")
    matrix = [
        [0, 1, 1, 0, 0],  # 1 → 2, 3
        [0, 0, 0, 1, 1],  # 2 → 4, 5
        [0, 0, 0, 0, 0],  # 3 (leaf)
        [0, 0, 0, 0, 0],  # 4 (leaf)
        [0, 0, 0, 0, 0]   # 5 (leaf)
    ]
    labels = [1, 2, 3, 4, 5]
    
    tree = BinaryTree.from_adjacency_matrix(matrix, labels)
    tree.print_tree()
    
    print("\n⚠️  Note: Matrix may not preserve left/right order")
    print("   For exact placement, use nested structure!")
    
    # Export to nested structure
    print("\n📋 Export to Nested Structure:")
    nested = tree.to_nested_structure()
    print(f"  {nested}")
    
    # Round-trip
    print("\n🔄 Round-trip: Nested → Tree → Nested")
    original = (1, [(2, [4, 5]), 3])
    tree_rt = BinaryTree.from_nested_structure(original)
    exported = tree_rt.to_nested_structure()
    print(f"  Original: {original}")
    print(f"  Exported: {exported}")
    print(f"  Match: {original == exported}")


def example_9_statistics():
    """
    Example 9: Tree statistics and analysis
    
    Demonstrates:
    - Complete statistics
    - Node counting
    - Height and depth
    - Tree properties
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Tree Statistics")
    print("=" * 80)
    
    # Create sample tree
    tree = BinaryTree.from_nested_structure((1, [(2, [4, 5]), (3, [6, 7])]))
    
    print("\n🌳 Tree:")
    tree.print_tree()
    
    print("\n📊 Statistics:")
    stats = tree.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n📐 Properties:")
    print(f"  Is Complete: {tree.is_complete()}")
    print(f"  Is Perfect: {tree.is_perfect()}")
    print(f"  Is Balanced: {tree.is_balanced()}")
    
    print("\n🔍 Individual Metrics:")
    print(f"  Total Nodes: {tree.get_node_count()}")
    print(f"  Leaf Nodes: {tree.get_leaf_count()}")
    print(f"  Inner Nodes: {tree.get_node_count() - tree.get_leaf_count()}")
    print(f"  Tree Height: {tree.get_height()}")
    print(f"  Edge Count: {tree.get_edge_count()}")


def example_10_comparison_with_tree():
    """
    Example 10: BinaryTree vs Tree comparison
    
    Demonstrates:
    - Inheritance relationship
    - Binary-specific features
    - Shared functionality
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 10: BinaryTree vs Tree")
    print("=" * 80)
    
    binary_tree = BinaryTree(1)
    binary_tree.insert_left(binary_tree.root, 2)
    binary_tree.insert_right(binary_tree.root, 3)
    
    print("\n🔷 BinaryTree Specialization:")
    print("  • Each node has at most 2 children (left, right)")
    print("  • Binary-specific methods: insert_left, insert_right")
    print("  • Binary traversals: inorder is meaningful")
    print("  • Binary properties: complete, perfect, balanced")
    
    print("\n🔶 Inherited from Tree:")
    print("  • All general tree operations")
    print("  • Graph theory methods")
    print("  • Import/export functionality")
    print("  • Generator-based iteration")
    print("  • Visualization")
    
    print("\n🌳 Binary Tree:")
    binary_tree.print_tree()
    
    print(f"\n📊 Can use inherited methods:")
    print(f"  Node Count: {binary_tree.get_node_count()}")
    print(f"  Height: {binary_tree.get_height()}")
    print(f"  Is Connected: {binary_tree.is_connected()}")
    print(f"  Has Cycle: {binary_tree.has_cycle()}")


def main():
    """Run all examples in sequence"""
    print("\n")
    print("🌲" * 40)
    print("BINARY TREE DATA STRUCTURE - COMPREHENSIVE DEMO")
    print("🌲" * 40)
    
    # Run all examples
    tree1 = example_1_manual_creation()
    example_2_binary_traversals(tree1)
    example_3_generators(tree1)
    example_4_binary_properties()
    tree2 = example_5_nested_structure()
    example_6_expression_trees()
    example_7_bst_sorting()
    example_8_import_export()
    example_9_statistics()
    example_10_comparison_with_tree()
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)
    print("\n📚 Key Takeaways:")
    print("  1. BinaryTree extends Tree with binary constraints")
    print("  2. Each node has at most 2 children (left, right)")
    print("  3. Use nested structure for easy tree creation")
    print("  4. Different traversals serve different purposes")
    print("  5. Binary trees have special properties (complete, perfect, balanced)")
    print("  6. Inorder traversal gives sorted output for BST")
    print("  7. Expression trees represent mathematical operations")
    print("\n")


if __name__ == "__main__":
    main()
