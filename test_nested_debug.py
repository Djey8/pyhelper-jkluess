"""Quick test of nested structure parsing"""

from pyhelper_jkluess.Complex.Trees import BinaryTree

# Test with simple structure
structure = (1, (2,), (3,))
print(f"Creating tree from: {structure}")

tree = BinaryTree.from_nested_structure(structure)
print(f"Root data: {tree.root.data}")
print(f"Root type: {type(tree.root.data)}")
print(f"Children count: {len(tree.root.children)}")
if tree.root.children:
    print(f"Children: {[c.data for c in tree.root.children]}")

print("\nTree structure:")
tree.print_tree()

print("\nPre-order traversal:")
print(tree.traverse_preorder())
