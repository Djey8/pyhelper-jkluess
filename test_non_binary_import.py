"""Test what happens when importing non-binary structures into BinaryTree"""

from pyhelper_jkluess.Complex.Trees import BinaryTree

print("=" * 60)
print("Testing Non-Binary Structures with BinaryTree")
print("=" * 60)

# Test 1: Nested structure with 3 children
print("\n1. Nested structure with 3 children:")
print("   Structure: (1, (2,), (3,), (4,))")
try:
    tree = BinaryTree.from_nested_structure((1, (2,), (3,), (4,)))
    print(f"   ✓ Tree created successfully!")
    print(f"   Root: {tree.root.data}")
    print(f"   Children count: {len(tree.root.children)}")
    print(f"   Children: {[c.data for c in tree.root.children]}")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

# Test 2: Adjacency matrix with 3 children
print("\n2. Adjacency matrix with node having 3 children:")
matrix = [
    [0, 1, 1, 1],  # Node 0 connects to 1, 2, 3 (3 children!)
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
labels = [1, 2, 3, 4]
print(f"   Matrix: {matrix[0]}")
try:
    tree = BinaryTree.from_adjacency_matrix(matrix, labels)
    print(f"   ✓ Tree created successfully!")
    print(f"   Root: {tree.root.data}")
    print(f"   Children count: {len(tree.root.children)}")
    print(f"   Children: {[c.data for c in tree.root.children]}")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

# Test 3: Adjacency list with 3 children
print("\n3. Adjacency list with node having 3 children:")
adj_list = {
    1: [2, 3, 4],  # 3 children!
    2: [],
    3: [],
    4: []
}
print(f"   Adj list: {adj_list}")
try:
    tree = BinaryTree.from_adjacency_list(adj_list, root=1)
    print(f"   ✓ Tree created successfully!")
    print(f"   Root: {tree.root.data}")
    print(f"   Children count: {len(tree.root.children)}")
    print(f"   Children: {[c.data for c in tree.root.children]}")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

# Test 4: Deep structure with mixed valid/invalid nodes
print("\n4. Deep structure - valid parent, invalid child:")
print("   Structure: (1, (2, (3,), (4,), (5,)), (6,))")
try:
    tree = BinaryTree.from_nested_structure((1, (2, (3,), (4,), (5,)), (6,)))
    print(f"   ✓ Tree created successfully!")
    print(f"   Root children: {[c.data for c in tree.root.children]}")
    if tree.root.left:
        print(f"   Left child ({tree.root.left.data}) children: {[c.data for c in tree.root.left.children]}")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
