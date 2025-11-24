"""Test LCRS (Left-Child Right-Sibling) conversion from Tree to BinaryTree"""

from pyhelper_jkluess.Complex.Trees import Tree, BinaryTree

print("=" * 70)
print("LCRS Conversion: Tree to BinaryTree")
print("=" * 70)

# Test 1: Tree with 3 children (needs LCRS)
print("\n1. General tree with 3 children (LCRS needed):")
tree = Tree("A")
b = tree.add_child(tree.root, "B")
c = tree.add_child(tree.root, "C")
d = tree.add_child(tree.root, "D")

print("\nOriginal Tree:")
tree.print_tree()
print(f"Root has {len(tree.root.children)} children: {[child.data for child in tree.root.children]}")

binary = tree.to_binary_tree()
print("\nConverted to BinaryTree (LCRS):")
binary.print_tree()

print("\nLCRS Structure:")
print(f"  Root: {binary.root.data}")
print(f"  Left (first child): {binary.root.left.data if binary.root.left else None}")
print(f"  Right (next sibling): {binary.root.right.data if binary.root.right else None}")
if binary.root.left and binary.root.left.right:
    print(f"  Left's right (sibling): {binary.root.left.right.data}")

# Test 2: Complex tree with mixed nodes
print("\n" + "=" * 70)
print("2. Complex tree with both binary and non-binary nodes:")
tree2 = Tree("Root")
a = tree2.add_child(tree2.root, "A")
b = tree2.add_child(tree2.root, "B")
c = tree2.add_child(tree2.root, "C")
d = tree2.add_child(tree2.root, "D")

# A has 2 children (already binary)
tree2.add_child(a, "A1")
tree2.add_child(a, "A2")

# B has 3 children (needs LCRS)
tree2.add_child(b, "B1")
tree2.add_child(b, "B2")
tree2.add_child(b, "B3")

print("\nOriginal Tree:")
tree2.print_tree()

binary2 = tree2.to_binary_tree()
print("\nConverted to BinaryTree (LCRS):")
binary2.print_tree()

print("\nTraversals of converted tree:")
print(f"Pre-order: {binary2.traverse_preorder()}")
print(f"In-order: {binary2.traverse_inorder()}")

# Test 3: Already binary tree
print("\n" + "=" * 70)
print("3. Tree that's already binary (2 children max):")
tree3 = Tree(1)
left = tree3.add_child(tree3.root, 2)
right = tree3.add_child(tree3.root, 3)
tree3.add_child(left, 4)
tree3.add_child(left, 5)

print("\nOriginal Tree (already binary):")
tree3.print_tree()

binary3 = tree3.to_binary_tree()
print("\nConverted to BinaryTree (structure preserved):")
binary3.print_tree()

print("\nComparison:")
print(f"  Original traversal: {tree3.traverse_preorder()}")
print(f"  Binary traversal:   {binary3.traverse_preorder()}")
print(f"  Match: {tree3.traverse_preorder() == binary3.traverse_preorder()}")

# Test 4: Single node
print("\n" + "=" * 70)
print("4. Single node tree:")
tree4 = Tree("X")
print("\nOriginal:")
tree4.print_tree()

binary4 = tree4.to_binary_tree()
print("\nConverted:")
binary4.print_tree()

# Test 5: Empty tree
print("\n" + "=" * 70)
print("5. Empty tree:")
tree5 = Tree()
binary5 = tree5.to_binary_tree()
print(f"Empty tree converted: {binary5.is_empty()}")

print("\n" + "=" * 70)
print("All LCRS conversion tests completed!")
print("=" * 70)
