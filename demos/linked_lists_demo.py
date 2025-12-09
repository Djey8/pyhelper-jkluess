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
    print("\n📥 Appending elements: 1, 2, 3")
    for val in [1, 2, 3]:
        ll.append(val)
    ll.print_list()
    
    # Prepend
    print("\n📥 Prepending 0:")
    ll.prepend(0)
    ll.print_list()
    
    # Insert at position
    print("\n📥 Inserting 1.5 at position 2:")
    ll.insert(2, 1.5)
    ll.print_list()
    
    # Delete
    print("\n❌ Deleting value 1.5:")
    ll.delete(1.5)
    ll.print_list()
    
    # Search
    print("\n🔍 Searching for value 2:")
    print(f"  Found: {ll.search(2)}")
    
    print(f"\n📊 List length: {ll.get_length()}")
    
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
    print("\n📥 Appending elements: A, B, C, D")
    for val in ['A', 'B', 'C', 'D']:
        dll.append(val)
    dll.print_list()
    
    # Forward and backward traversal
    print("\n→ Forward traversal:")
    dll.print_list()
    
    print("\n← Backward traversal:")
    dll.print_list_reverse()
    
    # Delete from middle
    print("\n❌ Deleting 'B' (easier with doubly linked list):")
    dll.delete('B')
    dll.print_list()
    
    # Insert after specific node
    print("\n📥 Inserting 'B2' after 'A':")
    dll.insert_after('A', 'B2')
    dll.print_list()
    
    print("\n💡 Doubly Linked List Advantages:")
    print("  • Bidirectional traversal")
    print("  • Easier deletion (no need to track previous)")
    print("  • Can traverse backwards")
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
    Example 5: Common operations
    
    Demonstrates:
    - Length calculation
    - Searching
    - Printing
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Common Operations")
    print("=" * 80)
    
    # Create sample list
    ll = LinkedList()
    data = [10, 20, 30, 40, 50]
    print(f"\n📥 Creating list with: {data}")
    for val in data:
        ll.append(val)
    ll.print_list()
    
    # Length
    print(f"\n📏 Length: {ll.get_length()}")
    
    # Search
    print("\n🔍 Searching:")
    for val in [30, 60]:
        found = ll.search(val)
        print(f"  {val}: {'Found' if found else 'Not found'}")
    
    # Delete multiple
    print("\n❌ Deleting 20 and 40:")
    ll.delete(20)
    ll.delete(40)
    ll.print_list()
    
    print(f"\n📏 New length: {ll.get_length()}")


def main():
    """Run all examples in sequence"""
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
