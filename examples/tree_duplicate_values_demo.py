"""
Demo: Creating Trees with Duplicate Node Values using from_nested_structure()

This example shows how to create trees where multiple nodes can have the same value,
which is useful for mathematical expression trees, parse trees, etc.
"""

from pyhelper_jkluess.Complex.Trees.tree import Tree

print("=" * 70)
print("TREE WITH DUPLICATE VALUES - NESTED STRUCTURE METHOD")
print("=" * 70)

# Example 1: Math Expression Tree
# Expression: (3 + 4) * 5 + 2 * 3
# Notice: Two different * nodes and two different 3 nodes
print("\n1. Math Expression Tree: (3 + 4) * 5 + 2 * 3")
print("-" * 70)

structure = ('+', [
    ('*', [
        ('+', [3, 4]),
        5
    ]),
    ('*', [2, 3])
])

math_tree = Tree.from_nested_structure(structure)
math_tree.print_tree()

print(f"\nTree Statistics:")
print(f"  Total nodes: {math_tree.get_node_count()}")
print(f"  Height: {math_tree.get_height()}")
print(f"  Leaf nodes (operands): {len(math_tree.get_leaves())}")
print(f"  Inner nodes (operators): {len(math_tree.get_inner_nodes())}")

# Export back to nested structure
exported_structure = math_tree.to_nested_structure()
print(f"\nExported structure: {exported_structure}")

# Verify round-trip works
math_tree_copy = Tree.from_nested_structure(exported_structure)
print(f"Round-trip successful: {math_tree.get_node_count()} nodes → export → import → {math_tree_copy.get_node_count()} nodes")

# Example 2: Parse Tree with Repeated Keywords
print("\n\n2. Code Parse Tree with Repeated 'if' statements")
print("-" * 70)

parse_tree_structure = ('program', [
    ('if', [
        'condition1',
        ('block', [
            ('if', [  # Nested if with same value as parent
                'condition2',
                'statement1'
            ]),
            'statement2'
        ])
    ]),
    ('if', [  # Another if at same level
        'condition3',
        'statement3'
    ])
])

parse_tree = Tree.from_nested_structure(parse_tree_structure)
parse_tree.print_tree()

print(f"\nTree has {parse_tree.get_node_count()} nodes including 3 distinct 'if' nodes")

# Example 3: Simple demonstration with all same values
print("\n\n3. Tree where ALL nodes have the same value 'X'")
print("-" * 70)

same_value_structure = ('X', [
    ('X', ['X', 'X']),
    ('X', ['X'])
])

same_tree = Tree.from_nested_structure(same_value_structure)
same_tree.print_tree()

print(f"\nAll {same_tree.get_node_count()} nodes have value 'X' but are distinct objects")

# Example 4: Comparison with adjacency list (can't handle duplicates)
print("\n\n4. Why adjacency list can't handle duplicate values")
print("-" * 70)

print("\n❌ This WON'T work with from_adjacency_list (duplicate keys):")
print("""
adjacency_list = {
    '+': ['*', '*'],    # Can't have duplicate key
    '*': ['+', 5],      # Which * is this?
    '+': [3, 4],        # Duplicate key overwrites!
    ...
}
""")

print("\n✅ But this WORKS with from_nested_structure:")
print("""
structure = ('+', [
    ('*', [('+', [3, 4]), 5]),
    ('*', [2, 3])
])
""")

# Example 5: Creating the same structure with unique labels (old way)
print("\n\n5. Alternative: Using unique labels (old approach)")
print("-" * 70)

# Old way: manually create unique labels
old_adj_list = {
    "+1": ["*1", "*2"],
    "*1": ["+2", "5a"],
    "+2": ["3a", "4a"],
    "3a": [],
    "4a": [],
    "5a": [],
    "*2": ["2b", "3b"],
    "2b": [],
    "3b": []
}

old_tree = Tree.from_adjacency_list(old_adj_list, "+1")
print("\nOld approach with unique labels:")
old_tree.print_tree()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Use from_nested_structure() when:
✓ You need multiple nodes with the same value
✓ Building math expression trees
✓ Creating parse trees
✓ Representing nested structures naturally

Use from_adjacency_list() when:
✓ All node values are unique
✓ You have data in adjacency list format
✓ Simpler representation for unique-valued trees
""")
