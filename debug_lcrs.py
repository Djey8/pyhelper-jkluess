"""Debug LCRS conversion issue"""

from pyhelper_jkluess.Complex.Trees import Tree

# Test case that's ACTUALLY failing - Root has 3 children
tree = Tree("Root")
a = tree.add_child(tree.root, "A")
b = tree.add_child(tree.root, "B")
c = tree.add_child(tree.root, "C")  # 3rd child!

# A has 2 children (binary)
a1 = tree.add_child(a, "A1")
a2 = tree.add_child(a, "A2")

# B has 3 children
b1 = tree.add_child(b, "B1")
b2 = tree.add_child(b, "B2")
b3 = tree.add_child(b, "B3")

print(f"A has {len(a.children)} children: {[c.data for c in a.children]}")
print(f"A1: {a1.data}, A2: {a2.data}")

print("\nTree preorder:", tree.traverse_preorder())

binary = tree.to_binary_tree()
print("Binary preorder:", binary.traverse_preorder())

# Where is B?
print("\nLooking for B:")
print(f"Root.right: {binary.root.right.data if binary.root.right else 'None'}")

if binary.root.right:
    print(f"B.left: {binary.root.right.left.data if binary.root.right.left else 'None'}")
    print(f"B.right: {binary.root.right.right.data if binary.root.right.right else 'None'}")

print(f"\nAll nodes in binary tree: {binary.traverse_preorder()}")
print(f"All nodes in original tree: {tree.traverse_preorder()}")
print(f"Match: {set(binary.traverse_preorder()) == set(tree.traverse_preorder())}")
