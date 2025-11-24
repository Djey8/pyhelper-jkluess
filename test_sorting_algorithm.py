"""Test the tree sorting algorithm (Baum-Sortier-Algorithmus)"""

from pyhelper_jkluess.Complex.Trees import BinaryTree

print("=" * 60)
print("Tree Sorting Algorithm (Baum-Sortier-Algorithmus)")
print("=" * 60)

# Test 1: Manual insertion
print("\n1. Manual sorted insertion:")
tree = BinaryTree()
values = [5, 3, 7, 1, 9, 4, 6]
print(f"   Inserting values: {values}")

for val in values:
    tree.insert_sorted(val)

print("\n   Tree structure:")
tree.print_tree()

print(f"\n   Pre-order:  {tree.traverse_preorder()}")
print(f"   In-order:   {tree.traverse_inorder()}  ← Sorted!")
print(f"   Post-order: {tree.traverse_postorder()}")

# Test 2: Using from_sorted_values
print("\n" + "=" * 60)
print("2. Using from_sorted_values():")
unsorted = [8, 3, 10, 1, 6, 14, 4, 7, 13]
print(f"   Input (unsorted): {unsorted}")

tree2 = BinaryTree.from_sorted_values(unsorted)

print("\n   Tree structure:")
tree2.print_tree()

print(f"\n   Sorted output (In-order): {tree2.traverse_inorder()}")
print(f"   Verification: {tree2.traverse_inorder() == sorted(unsorted)}")

# Test 3: With duplicates (should be ignored)
print("\n" + "=" * 60)
print("3. With duplicates (ignored):")
values_with_dupes = [5, 3, 7, 3, 5, 9, 1, 7]
print(f"   Input: {values_with_dupes}")

tree3 = BinaryTree.from_sorted_values(values_with_dupes)
result = tree3.traverse_inorder()

print(f"   Output: {result}")
print(f"   Unique values: {sorted(set(values_with_dupes))}")
print(f"   Match: {result == sorted(set(values_with_dupes))}")

# Test 4: Different data types
print("\n" + "=" * 60)
print("4. Sorting strings:")
words = ["dog", "cat", "elephant", "ant", "bear"]
print(f"   Input: {words}")

tree4 = BinaryTree.from_sorted_values(words)
result = tree4.traverse_inorder()

print(f"   Sorted: {result}")
print(f"   Verification: {result == sorted(words)}")

print("\n" + "=" * 60)
print("Algorithm verification complete!")
print("=" * 60)
