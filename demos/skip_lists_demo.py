"""
Skip Lists - Comprehensive Demo

This demo showcases Skip List data structures:
- SkipList (deterministic)
- ProbabilisticSkipList (randomized)

Topics covered:
1. Skip list concept and structure
2. Insertion operations
3. Search operations
4. Deletion operations
5. Probabilistic vs deterministic
6. Performance characteristics

Author: PyHelper JKluess
Date: December 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhelper_jkluess.Complex.SkipLists.skiplist import SkipList
from pyhelper_jkluess.Complex.SkipLists.probabilisticskiplist import ProbabilisticSkipList


def example_1_skiplist_basics():
    """
    Example 1: SkipList basics
    
    Demonstrates:
    - Creating skip list
    - Inserting key-value pairs
    - Searching
    - Deletion
    """
    print("=" * 80)
    print("EXAMPLE 1: SkipList Basics")
    print("=" * 80)
    
    sl = SkipList()
    print("✓ Created SkipList (deterministic)")
    
    # Insert key-value pairs
    print("\n📥 Inserting key-value pairs:")
    pairs = [(1, "one"), (2, "two"), (3, "three"), (5, "five"), (7, "seven")]
    for key, value in pairs:
        sl.insert(key, value)
        print(f"  {key} → {value}")
    
    # Search
    print("\n🔍 Searching:")
    for key in [2, 4, 5]:
        result = sl.search(key)
        if result:
            print(f"  Key {key}: Found → {result}")
        else:
            print(f"  Key {key}: Not found")
    
    # Delete
    print("\n❌ Deleting key 2:")
    sl.delete(2)
    result = sl.search(2)
    print(f"  Search for 2: {'Found' if result else 'Not found'}")
    
    print("\n💡 SkipList:")
    print("  • Ordered data structure")
    print("  • Multiple levels for fast search")
    print("  • O(log n) search, insert, delete (average)")
    print("  • Alternative to balanced trees")
    
    return sl


def example_2_probabilistic_skiplist():
    """
    Example 2: ProbabilisticSkipList
    
    Demonstrates:
    - Randomized level assignment
    - Add, find, remove operations
    - Display functionality
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Probabilistic SkipList")
    print("=" * 80)
    
    psl = ProbabilisticSkipList()
    print("✓ Created ProbabilisticSkipList (randomized levels)")
    
    # Add elements
    print("\n📥 Adding elements: 1, 2, 3, 5, 8, 13")
    for val in [1, 2, 3, 5, 8, 13]:
        psl.add(val)
        print(f"  Added {val}")
    
    # Display
    print("\n📊 Skip list structure:")
    display = psl.display()
    print(f"  {display}")
    
    # Find elements
    print("\n🔍 Finding elements:")
    for val in [3, 7, 13]:
        found = psl.find(val)
        print(f"  {val}: {'Found' if found else 'Not found'}")
    
    # Remove
    print("\n❌ Removing 3:")
    psl.remove(3)
    display = psl.display()
    print(f"  After removal: {display}")
    
    print("\n💡 Probabilistic SkipList:")
    print("  • Randomly assigns levels")
    print("  • Simpler implementation")
    print("  • Good average performance")
    print("  • No rebalancing needed")
    
    return psl


def example_3_performance():
    """
    Example 3: Performance characteristics
    
    Demonstrates:
    - Time complexity
    - Space complexity
    - Comparison with other structures
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Performance Characteristics")
    print("=" * 80)
    
    print("\n⏱️  Time Complexity:")
    print("  Operation     Average    Worst Case")
    print("  ───────────   ────────   ──────────")
    print("  Search        O(log n)   O(n)")
    print("  Insert        O(log n)   O(n)")
    print("  Delete        O(log n)   O(n)")
    
    print("\n💾 Space Complexity:")
    print("  • O(n) average")
    print("  • Each node can have multiple forward pointers")
    print("  • Expected number of pointers: n * (1 + 1/2 + 1/4 + ...) ≈ 2n")
    
    print("\n📊 Comparison with Other Structures:")
    
    print("\n  vs. Sorted Array:")
    print("    ✓ Faster insert/delete (no shifting needed)")
    print("    ✗ Slower access by index")
    
    print("\n  vs. Balanced BST (AVL, Red-Black):")
    print("    ✓ Simpler implementation (no rotations)")
    print("    ✓ Probabilistic guarantee (no rebalancing)")
    print("    ≈ Similar average performance")
    
    print("\n  vs. Hash Table:")
    print("    ✓ Maintains sorted order")
    print("    ✓ Range queries possible")
    print("    ✗ Slower than hash table for single lookups")


def example_4_use_cases():
    """
    Example 4: Real-world use cases
    
    Demonstrates:
    - When to use skip lists
    - Practical applications
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Use Cases")
    print("=" * 80)
    
    print("\n🎯 When to Use Skip Lists:")
    
    print("\n  1. Ordered Data with Frequent Updates:")
    print("     • Database indices")
    print("     • In-memory sorted sets")
    print("     • Priority queues with order")
    
    print("\n  2. Range Queries:")
    print("     • Find all elements between x and y")
    print("     • Sorted iteration")
    print("     • Rank queries")
    
    print("\n  3. Concurrent Access:")
    print("     • Lock-free skip lists possible")
    print("     • Better for concurrent operations than trees")
    print("     • Used in some concurrent databases")
    
    print("\n  4. Simple Implementation Preferred:")
    print("     • Easier than balanced trees")
    print("     • No complex rotations")
    print("     • Randomization simplifies logic")
    
    print("\n📚 Real-World Examples:")
    print("  • Redis: Uses skip lists for sorted sets")
    print("  • LevelDB/RocksDB: MemTable implementation")
    print("  • Lucene: Term dictionary in some versions")
    print("  • Some in-memory databases")


def example_5_levels_concept():
    """
    Example 5: Understanding levels
    
    Demonstrates:
    - Multi-level structure
    - Express lanes concept
    - How levels speed up search
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Understanding Skip List Levels")
    print("=" * 80)
    
    print("\n🏗️  Skip List Structure:")
    print("\n  Level 3:  Head -----------------> 10 ---------> Tail")
    print("  Level 2:  Head ------> 5 -------> 10 -> 15 ----> Tail")
    print("  Level 1:  Head -> 2 -> 5 -> 7 -> 10 -> 15 -> 20 -> Tail")
    print("  Level 0:  Head -> 2 -> 5 -> 7 -> 10 -> 15 -> 20 -> Tail")
    
    print("\n🚗 Highway Analogy:")
    print("  • Level 0 = Local roads (all nodes)")
    print("  • Level 1 = Minor highways (every ~2nd node)")
    print("  • Level 2 = Major highways (every ~4th node)")
    print("  • Level 3 = Express lanes (every ~8th node)")
    
    print("\n🔍 Search Process:")
    print("  1. Start at highest level")
    print("  2. Move forward while next < target")
    print("  3. Drop down one level")
    print("  4. Repeat until found or level 0")
    
    print("\n📈 Why It's Fast:")
    print("  • Skip many nodes at high levels")
    print("  • Binary search-like behavior")
    print("  • Average O(log n) comparisons")
    print("  • Probabilistic guarantee")
    
    # Demonstrate search with example
    psl = ProbabilisticSkipList()
    for val in [2, 5, 7, 10, 15, 20]:
        psl.add(val)
    
    print("\n🔍 Example: Searching for 15")
    print("  Path might be: Head → 10 (level 3) → 15 (level 2) → Found")
    print(f"  Result: {psl.find(15)}")


def example_6_deterministic_vs_probabilistic():
    """
    Example 6: Deterministic vs Probabilistic
    
    Demonstrates:
    - Differences between implementations
    - Trade-offs
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Deterministic vs Probabilistic Skip Lists")
    print("=" * 80)
    
    print("\n🎲 Deterministic SkipList:")
    print("  • Predefined level structure")
    print("  • Key-value pairs")
    print("  • More predictable")
    print("  • Suitable for small datasets")
    
    print("\n🎲 Probabilistic SkipList:")
    print("  • Random level assignment (coin flip)")
    print("  • Typically p=0.5 or p=0.25")
    print("  • Simpler implementation")
    print("  • Better for large datasets")
    print("  • Self-balancing through randomization")
    
    print("\n⚖️  Trade-offs:")
    print("\n  Deterministic:")
    print("    ✓ Predictable structure")
    print("    ✓ Good for testing/debugging")
    print("    ✗ May need manual balancing")
    
    print("\n  Probabilistic:")
    print("    ✓ No balancing needed")
    print("    ✓ Simple to implement")
    print("    ✓ Good average case")
    print("    ✗ Worst case can be bad (rare)")
    print("    ✗ Performance varies with random seed")


def main():
    """Run all examples in sequence"""
    import sys
    import io
    # Fix Windows console encoding for emoji support
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n")
    print("⏩" * 40)
    print("SKIP LISTS - COMPREHENSIVE DEMO")
    print("⏩" * 40)
    
    # Run all examples
    example_1_skiplist_basics()
    example_2_probabilistic_skiplist()
    example_3_performance()
    example_4_use_cases()
    example_5_levels_concept()
    example_6_deterministic_vs_probabilistic()
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)
    print("\n📚 Key Takeaways:")
    print("  1. Skip lists are probabilistic data structures")
    print("  2. Multi-level structure enables fast search (O(log n))")
    print("  3. Simpler than balanced trees (no rotations)")
    print("  4. Maintains sorted order")
    print("  5. Good for concurrent access")
    print("  6. Used in Redis, LevelDB, and other systems")
    print("  7. Probabilistic version uses randomization for levels")
    print("  8. Express lanes analogy: higher levels skip more nodes")
    print("\n")


if __name__ == "__main__":
    main()
