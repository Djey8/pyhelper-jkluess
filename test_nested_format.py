"""Test proper nested structure format with 3 children"""

from pyhelper_jkluess.Complex.Trees import BinaryTree

print("Testing correct nested structure format with 3 children:")
print("-" * 60)

# Correct format: (value, [child1, child2, child3])
structure = (1, [(2,), (3,), (4,)])  # Root 1 with 3 children
print(f"Structure: {structure}")

try:
    tree = BinaryTree.from_nested_structure(structure)
    print(f"✓ Tree created!")
    print(f"Root: {tree.root.data}")
    print(f"Children: {[c.data for c in tree.root.children]}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
