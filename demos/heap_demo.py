"""
Heap Data Structure - Comprehensive Demo

This demo showcases the complete functionality of the Heap class,
including min-heap, max-heap, heap sort, and various heap operations.

Topics covered:
1. Min-Heap operations
2. Max-Heap operations
3. Heap insertion and extraction
4. Heap sort (ascending and descending)
5. Heapify operation
6. Heap with different data types
7. Heap properties and BinaryTree inheritance
8. Understanding partial order

Author: PyHelper JKluess
Date: December 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhelper_jkluess.Complex.Trees.heap import Heap, HeapType, heap_sort, heapify


def example_1_min_heap():
    """
    Example 1: Min-Heap operations
    
    Demonstrates:
    - Creating min-heap
    - Inserting elements
    - Peeking at minimum
    - Tree structure visualization
    """
    print("=" * 80)
    print("EXAMPLE 1: Min-Heap Operations")
    print("=" * 80)
    
    # Create min-heap
    min_heap = Heap(HeapType.MIN)
    print("✓ Created Min-Heap")
    
    # Insert values
    values = [10, 12, 13, 14, 11, 15, 16]
    print(f"\n📥 Inserting values: {values}")
    for val in values:
        min_heap.heap_insert(val)
        print(f"  Inserted {val}, heap: {min_heap}")
    
    print(f"\n📊 Min-Heap Properties:")
    print(f"  Root (minimum): {min_heap.heap_peek()}")
    print(f"  Size: {min_heap.heap_size()}")
    print(f"  Array representation: {min_heap.to_array()}")
    
    print("\n🌳 Tree Structure:")
    min_heap.print_tree()
    
    print("\n💡 Min-Heap Property:")
    print("  • Parent ≤ Children")
    print("  • Root is minimum element")
    print("  • Peek: O(1), Insert: O(log n), Extract: O(log n)")
    
    return min_heap


def example_2_max_heap():
    """
    Example 2: Max-Heap operations
    
    Demonstrates:
    - Creating max-heap
    - Contrast with min-heap
    - Maximum element access
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Max-Heap Operations")
    print("=" * 80)
    
    # Create max-heap
    max_heap = Heap(HeapType.MAX)
    print("✓ Created Max-Heap")
    
    # Insert same values as min-heap
    values = [10, 12, 13, 14, 11, 15, 16]
    print(f"\n📥 Inserting values: {values}")
    for val in values:
        max_heap.heap_insert(val)
    
    print(f"\n📊 Max-Heap Properties:")
    print(f"  Root (maximum): {max_heap.heap_peek()}")
    print(f"  Size: {max_heap.heap_size()}")
    print(f"  Array representation: {max_heap.to_array()}")
    
    print("\n🌳 Tree Structure:")
    max_heap.print_tree()
    
    print("\n💡 Max-Heap Property:")
    print("  • Parent ≥ Children")
    print("  • Root is maximum element")
    print("  • Same time complexity as min-heap")
    
    return max_heap


def example_3_extraction(min_heap):
    """
    Example 3: Extracting elements
    
    Demonstrates:
    - Removing root element
    - Heap restructuring
    - Sorted output
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Extracting from Min-Heap")
    print("=" * 80)
    
    print("\n🌳 Original Heap:")
    print(f"  {min_heap}")
    
    print("\n📤 Extracting all elements:")
    extracted = []
    while not min_heap.is_heap_empty():
        value = min_heap.heap_extract()
        extracted.append(value)
        print(f"  Extracted: {value}, Remaining: {min_heap if not min_heap.is_heap_empty() else 'empty'}")
    
    print(f"\n📋 Extracted order: {extracted}")
    print("  Note: Elements come out in sorted order!")
    
    print("\n💡 Heap Extract:")
    print("  1. Remove and return root")
    print("  2. Move last element to root")
    print("  3. Heapify down to restore heap property")
    print("  4. Time complexity: O(log n)")


def example_4_heap_sort_ascending():
    """
    Example 4: Heap sort (ascending)
    
    Demonstrates:
    - Sorting using heap
    - In-place sorting
    - Time complexity
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Heap Sort (Ascending)")
    print("=" * 80)
    
    unsorted = [5, 3, 7, 1, 9, 2, 8, 4, 6]
    print(f"\n📋 Unsorted array: {unsorted}")
    
    sorted_asc = heap_sort(unsorted)
    
    print(f"📋 Sorted (ascending): {sorted_asc}")
    
    print("\n🔄 Heap Sort Algorithm:")
    print("  1. Build max-heap from array")
    print("  2. Repeatedly extract max (swap with last)")
    print("  3. Heapify remaining elements")
    print("  4. Result: sorted in ascending order")
    
    print("\n⏱️  Time Complexity: O(n log n)")
    print("   Space Complexity: O(1) - in-place")


def example_5_heap_sort_descending():
    """
    Example 5: Heap sort (descending)
    
    Demonstrates:
    - Reverse sorting
    - Using reverse parameter
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Heap Sort (Descending)")
    print("=" * 80)
    
    unsorted = [5, 3, 7, 1, 9, 2, 8, 4, 6]
    print(f"\n📋 Unsorted array: {unsorted}")
    
    sorted_desc = heap_sort(unsorted, reverse=True)
    
    print(f"📋 Sorted (descending): {sorted_desc}")
    
    print("\n💡 For descending order:")
    print("  • Use heap_sort(array, reverse=True)")
    print("  • Or use min-heap instead of max-heap")


def example_6_heapify():
    """
    Example 6: Heapify operation
    
    Demonstrates:
    - Building heap from existing data
    - Efficient bulk insertion
    - Bottom-up heap construction
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Heapify Operation")
    print("=" * 80)
    
    data = [15, 10, 20, 8, 12, 25, 6]
    print(f"\n📋 Original data: {data}")
    
    # Heapify into min-heap
    heap = heapify(data, HeapType.MIN)
    
    print(f"\n📊 After heapify (min-heap):")
    print(f"  Array: {heap.to_array()}")
    print(f"  Root: {heap.heap_peek()}")
    
    print("\n🌳 Tree Structure:")
    heap.print_tree()
    
    print("\n💡 Heapify:")
    print("  • Converts array to heap in-place")
    print("  • More efficient than n insertions")
    print("  • Time complexity: O(n) vs O(n log n)")
    print("  • Bottom-up construction")


def example_7_duplicates():
    """
    Example 7: Heap with duplicate values
    
    Demonstrates:
    - Handling duplicates
    - Sorting with duplicates
    - Stable ordering
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Heap with Duplicates")
    print("=" * 80)
    
    data_with_dupes = [5, 3, 5, 1, 3, 7, 1, 9, 3]
    print(f"\n📋 Data with duplicates: {data_with_dupes}")
    
    # Create heap
    heap = heapify(data_with_dupes, HeapType.MIN)
    print(f"\n📊 Min-Heap: {heap.to_array()}")
    
    # Sort
    sorted_data = heap_sort(data_with_dupes.copy())
    print(f"📋 Sorted: {sorted_data}")
    
    print("\n💡 Duplicates:")
    print("  • Heaps handle duplicates naturally")
    print("  • No special logic needed")
    print("  • All duplicates preserved in output")


def example_8_string_heap():
    """
    Example 8: Heap with strings
    
    Demonstrates:
    - Heap works with any comparable type
    - Alphabetical ordering
    - Lexicographic comparison
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Heap with Strings")
    print("=" * 80)
    
    # Create min-heap for strings
    string_heap = Heap(HeapType.MIN)
    words = ["banana", "apple", "cherry", "date", "elderberry"]
    
    print(f"\n📋 Inserting words: {words}")
    for word in words:
        string_heap.heap_insert(word)
    
    print(f"\n📊 Min-Heap (alphabetical):")
    print(f"  Root: {string_heap.heap_peek()}")
    print(f"  Heap: {string_heap.to_array()}")
    
    print("\n🌳 Tree Structure:")
    string_heap.print_tree()
    
    # Extract in order
    print("\n📤 Extracting in alphabetical order:")
    while not string_heap.is_heap_empty():
        print(f"  {string_heap.heap_extract()}")
    
    print("\n💡 Heaps work with any comparable type:")
    print("  • Numbers (int, float)")
    print("  • Strings (lexicographic)")
    print("  • Custom objects (with __lt__, __gt__)")


def example_9_binary_tree_inheritance():
    """
    Example 9: BinaryTree inheritance
    
    Demonstrates:
    - Heap inherits from BinaryTree
    - Access to BinaryTree methods
    - Additional heap-specific properties
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 9: BinaryTree Inheritance")
    print("=" * 80)
    
    # Create heap
    heap = Heap(HeapType.MIN, [10, 12, 13, 14, 11, 15, 16])
    
    print("\n📊 Heap (inherits from BinaryTree):")
    heap.print_tree()
    
    print(f"\n🔷 Heap-specific methods:")
    print(f"  heap_size(): {heap.heap_size()}")
    print(f"  heap_peek(): {heap.heap_peek()}")
    print(f"  is_heap_empty(): {heap.is_heap_empty()}")
    print(f"  to_array(): {heap.to_array()}")
    
    print(f"\n🔶 Inherited from BinaryTree:")
    print(f"  get_node_count(): {heap.get_node_count()}")
    print(f"  get_height(): {heap.get_height()}")
    print(f"  is_complete(): {heap.is_complete()}")
    print(f"  is_perfect(): {heap.is_perfect()}")
    print(f"  get_leaf_count(): {heap.get_leaf_count()}")
    
    print(f"\n🔶 Inherited from Tree:")
    print(f"  traverse_preorder(): {heap.traverse_preorder()}")
    print(f"  traverse_levelorder(): {heap.traverse_levelorder()}")
    print(f"  get_statistics(): {list(heap.get_statistics().keys())}")
    
    print("\n💡 Heaps are always:")
    print("  • Complete binary trees")
    print("  • NOT perfect (unless size = 2^k - 1)")
    print("  • NOT balanced (only complete)")


def example_10_partial_order():
    """
    Example 10: Understanding partial order
    
    Demonstrates:
    - Heap provides partial order
    - Compromise between unordered and sorted
    - Fast operations
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Partial Order in Heaps")
    print("=" * 80)
    
    print("\n📊 Data Organization Spectrum:")
    print("\n  1. Unordered Array: [5, 3, 7, 1, 9]")
    print("     • Insert: O(1)")
    print("     • Find min: O(n)")
    print("     • Extract min: O(n)")
    
    print("\n  2. Heap (Partial Order): [1, 3, 7, 5, 9]")
    print("     • Insert: O(log n)")
    print("     • Find min: O(1)")
    print("     • Extract min: O(log n)")
    print("     • Parent-child relationship maintained")
    
    print("\n  3. Sorted Array: [1, 3, 5, 7, 9]")
    print("     • Insert: O(n) - must maintain order")
    print("     • Find min: O(1)")
    print("     • Extract min: O(n) - must shift elements")
    
    print("\n💡 Heap = Perfect Balance:")
    print("  • Faster than sorted array for insertions")
    print("  • Faster than unordered array for min/max")
    print("  • Ideal for priority queues")
    
    # Demonstrate with actual heap
    heap = Heap(HeapType.MIN, [5, 3, 7, 1, 9])
    print(f"\n🌳 Min-Heap structure:")
    heap.print_tree()
    print(f"  Array: {heap.to_array()}")
    print("  Note: Not fully sorted, but min is at root!")


def example_11_use_cases():
    """
    Example 11: Real-world use cases
    
    Demonstrates:
    - Priority queues
    - Top-K problems
    - Median finding
    - Event scheduling
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 11: Real-World Use Cases")
    print("=" * 80)
    
    print("\n🎯 Common Use Cases:")
    
    print("\n  1. Priority Queue:")
    print("     • OS task scheduling")
    print("     • Emergency room patient prioritization")
    print("     • Network packet routing")
    
    print("\n  2. Top-K Problems:")
    print("     • Find K largest elements")
    print("     • Find K smallest elements")
    print("     • Top K frequent items")
    
    print("\n  3. Heap Sort:")
    print("     • O(n log n) sorting")
    print("     • In-place sorting")
    print("     • No worst-case O(n²)")
    
    print("\n  4. Graph Algorithms:")
    print("     • Dijkstra's shortest path")
    print("     • Prim's minimum spanning tree")
    print("     • A* pathfinding")
    
    print("\n  5. Streaming Data:")
    print("     • Running median")
    print("     • Online statistics")
    print("     • Event simulation")
    
    # Example: Top-K problem
    print("\n📋 Example: Find 3 smallest numbers")
    data = [7, 10, 4, 3, 20, 15, 1, 12]
    print(f"  Data: {data}")
    
    # Use max-heap of size k
    sorted_data = heap_sort(data)
    top_3 = sorted_data[:3]
    print(f"  Top 3 smallest: {top_3}")


def main():
    """Run all examples in sequence"""
    import sys
    import io
    # Fix Windows console encoding for emoji support
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n")
    print("🏛️" * 40)
    print("HEAP DATA STRUCTURE - COMPREHENSIVE DEMO")
    print("🏔️ " * 40)
    
    # Run all examples
    min_heap = example_1_min_heap()
    max_heap = example_2_max_heap()
    
    # Recreate min_heap for extraction example
    min_heap = Heap(HeapType.MIN, [10, 12, 13, 14, 11, 15, 16])
    example_3_extraction(min_heap)
    
    example_4_heap_sort_ascending()
    example_5_heap_sort_descending()
    example_6_heapify()
    example_7_duplicates()
    example_8_string_heap()
    example_9_binary_tree_inheritance()
    example_10_partial_order()
    example_11_use_cases()
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)
    print("\n📚 Key Takeaways:")
    print("  1. Heaps provide partial order (compromise between unordered and sorted)")
    print("  2. Min-Heap: parent ≤ children, root is minimum")
    print("  3. Max-Heap: parent ≥ children, root is maximum")
    print("  4. Heap operations: Insert O(log n), Extract O(log n), Peek O(1)")
    print("  5. Heap sort: O(n log n) time, O(1) space")
    print("  6. Heaps are always complete binary trees")
    print("  7. Perfect for priority queues and top-K problems")
    print("  8. Heapify: O(n) to build heap from array")
    print("\n")


if __name__ == "__main__":
    main()
