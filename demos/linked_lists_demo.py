"""
Linked Lists - Comprehensive Demo

This demo showcases different types of linked lists:
- Singly Linked List
- Doubly Linked List  
- Circular Linked List

Topics covered:
1. Creating and manipulating linked lists
2. Insertion operations (append, prepend, insert at position)
3. Deletion operations
4. Traversal and search
5. Reversing lists
6. List properties

Author: PyHelper JKluess
Date: December 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhelper_jkluess.Basic.Lists.linked_list import LinkedList
from pyhelper_jkluess.Basic.Lists.double_linked_list import DoubleLinkedList
from pyhelper_jkluess.Basic.Lists.circular_linked_list import CircularLinkedList


def example_1_singly_linked_list():
    """
    Example 1: Singly Linked List
    
    Demonstrates:
    - Append, prepend, insert
    - Delete operations
    - Traversal
    """
    print("=" * 80)
    print("EXAMPLE 1: Singly Linked List")
    print("=" * 80)
    
    ll = LinkedList()
    print("✓ Created empty singly linked list")
    
    # Append elements
    print("\n📥 Appending elements: 1, 2, 3, 4, 5")
    for val in [1, 2, 3, 4, 5]:
        ll.append(val)
    ll.print_list()
    
    # Remove by index
    print("\n❌ Removing element at index 2 (value 3):")
    ll.remove(2)
    ll.print_list()
    
    # Remove head
    print("\n❌ Removing element at index 0 (head):")
    ll.remove(0)
    ll.print_list()
    
    # Append more
    print("\n📥 Appending 6 and 7:")
    ll.append(6)
    ll.append(7)
    ll.print_list()
    
    print("\n💡 LinkedList operations:")
    print("  • append(data) - Add to end (O(n))")
    print("  • remove(index) - Remove by index (O(n))")
    print("  • print_list() - Display all elements")
    
    return ll


def example_2_doubly_linked_list():
    """
    Example 2: Doubly Linked List
    
    Demonstrates:
    - Bidirectional traversal
    - Easier deletion
    - Reverse traversal
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Doubly Linked List")
    print("=" * 80)
    
    dll = DoubleLinkedList()
    print("✓ Created empty doubly linked list")
    
    # Append elements
    print("\n📥 Appending elements: A, B, C, D, E")
    for val in ['A', 'B', 'C', 'D', 'E']:
        dll.append(val)
    dll.print_list()
    
    # Forward and backward traversal
    print("\n→ Forward traversal:")
    dll.print_list()
    
    print("\n← Backward traversal:")
    dll.print_list_backwards()
    
    # Remove from middle (index 2 = 'C')
    print("\n❌ Removing element at index 2 ('C'):")
    dll.remove(2)
    dll.print_list()
    
    # Remove head (index 0)
    print("\n❌ Removing element at index 0 (head 'A'):")
    dll.remove(0)
    dll.print_list()
    
    # Show backward again
    print("\n← Backward after deletions:")
    dll.print_list_backwards()
    
    print("\n💡 Doubly Linked List Advantages:")
    print("  • Bidirectional traversal")
    print("  • Can traverse forwards and backwards")
    print("  • Each node has prev and next pointers")
    print("  • Extra space for prev pointer")
    
    return dll


def example_3_circular_linked_list():
    """
    Example 3: Circular Linked List
    
    Demonstrates:
    - Circular structure
    - No null termination
    - Useful for round-robin
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Circular Linked List")
    print("=" * 80)
    
    cll = CircularLinkedList()
    print("✓ Created empty circular linked list")
    
    # Append elements
    print("\n📥 Appending elements: 1, 2, 3, 'a'")
    for val in [1, 2, 3, 'a']:
        cll.append(val)
    cll.print_list()
    
    # Delete element
    print("\n❌ Deleting value 2:")
    cll.delete(2)
    cll.print_list()
    
    # Append more
    print("\n📥 Appending value 5:")
    cll.append(5)
    cll.print_list()
    
    print("\n💡 Circular Linked List:")
    print("  • Last node points back to first")
    print("  • No null/None at end")
    print("  • Useful for: round-robin scheduling, circular buffers")
    print("  • Be careful with traversal (can loop forever!)")
    
    return cll


def example_4_comparison():
    """
    Example 4: Comparison of linked list types
    
    Demonstrates:
    - When to use each type
    - Trade-offs
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Linked List Comparison")
    print("=" * 80)
    
    print("\n📊 Singly Linked List:")
    print("  ✓ Memory efficient (one pointer per node)")
    print("  ✓ Simple implementation")
    print("  ✗ One-way traversal only")
    print("  ✗ Deletion requires tracking previous node")
    print("  Use for: Stacks, simple forward-only lists")
    
    print("\n📊 Doubly Linked List:")
    print("  ✓ Bidirectional traversal")
    print("  ✓ Easier deletion (O(1) with node reference)")
    print("  ✓ Can navigate backwards")
    print("  ✗ More memory (two pointers per node)")
    print("  ✗ More complex implementation")
    print("  Use for: Browser history, undo/redo, LRU cache")
    
    print("\n📊 Circular Linked List:")
    print("  ✓ No null/None termination")
    print("  ✓ Any node can be starting point")
    print("  ✓ Natural for cyclic operations")
    print("  ✗ Must track when to stop traversing")
    print("  ✗ Can accidentally loop forever")
    print("  Use for: Round-robin scheduling, circular buffers, music playlists")
    
    print("\n⚖️  Time Complexities (all types):")
    print("  • Access by index: O(n)")
    print("  • Search: O(n)")
    print("  • Insert at head: O(1)")
    print("  • Insert at tail: O(1) with tail pointer, O(n) without")
    print("  • Delete: O(n) to find, then O(1) to remove")


def example_5_operations():
    """
    Example 5: Common operations demonstration
    
    Demonstrates:
    - Building lists
    - Removing elements
    - Printing
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Common Operations Summary")
    print("=" * 80)
    
    # Create sample list
    ll = LinkedList()
    data = [10, 20, 30, 40, 50]
    print(f"\n📥 Creating list with: {data}")
    for val in data:
        ll.append(val)
    ll.print_list()
    
    # Remove operations
    print("\n❌ Removing elements at index 1, then index 2:")
    ll.remove(1)  # Removes 20
    ll.print_list()
    ll.remove(2)  # Removes 40 (now at index 2)
    ll.print_list()
    
    print("\n📥 Appending 60 and 70:")
    ll.append(60)
    ll.append(70)
    ll.print_list()
    
    print("\n💡 Summary:")
    print("  • LinkedList: append(), remove(index), print_list()")
    print("  • DoubleLinkedList: same + print_list_backwards()")
    print("  • CircularLinkedList: append(), delete(value), print_list()")


def main():
    """Run all examples in sequence"""
    import sys
    import io
    # Fix Windows console encoding for emoji support
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n")
    print("🔗" * 40)
    print("LINKED LISTS - COMPREHENSIVE DEMO")
    print("🔗" * 40)
    
    # Run all examples
    example_1_singly_linked_list()
    example_2_doubly_linked_list()
    example_3_circular_linked_list()
    example_4_comparison()
    example_5_operations()
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)
    print("\n📚 Key Takeaways:")
    print("  1. Linked lists provide dynamic size (no fixed capacity)")
    print("  2. Singly: Simple, one-way traversal")
    print("  3. Doubly: Bidirectional, easier deletion")
    print("  4. Circular: No end, useful for round-robin")
    print("  5. Insert/delete at head is O(1)")
    print("  6. Access by index is O(n) - slower than arrays")
    print("  7. No wasted space (unlike arrays with unused capacity)")
    print("\n")


if __name__ == "__main__":
    main()
