"""
Heap Data Structure Implementation

A heap is a specialized binary tree-based data structure that satisfies the heap property.
This module provides both Min-Heap and Max-Heap implementations, as well as heap sort.

Definition 5.10 (Heap):
A heap is a binary, ordered, vertex-valued rooted tree with the following properties:
1. The smallest (largest) value of a subtree is located in its root (Min-Heap, Max-Heap).
2. If h is the height of the heap, all nodes at levels 0 to h - 2 have the same node degree of 2.
3. If the last level is not fully occupied, the nodes are arranged continuously from left to right
   without gaps.

Properties:
- The root of the tree contains the smallest (largest) value.
- Every subtree of a heap is itself a heap.

Data Ordering in Heaps:
- In a heap, data is "partially" ordered compared to a list.
- This partial ordering is a compromise between:
  1. Completely unordered storage with fast data storage and expensive data search.
  2. Completely ordered storage with expensive data storage and fast data search.

Index Mapping:
In a heap, pointer operations can be replaced by index operations:
- Index 0: Root node
- For node at index i:
  - Left child: 2*i + 1
  - Right child: 2*i + 2
  - Parent node: (i-1) // 2
- Numbered level by level, from left to right
"""

from typing import List, Any, Optional
from enum import Enum

try:
    from .binary_tree import BinaryTree, BinaryNode
except ImportError:
    from binary_tree import BinaryTree, BinaryNode


class HeapType(Enum):
    """Enum for heap types"""
    MIN = "min"
    MAX = "max"


class Heap(BinaryTree):
    """
    A heap implementation extending BinaryTree, supporting both Min-Heap and Max-Heap.
    
    In a Min-Heap, the parent is smaller than or equal to its children.
    In a Max-Heap, the parent is greater than or equal to its children.
    
    The heap uses an array-based representation where for element at index i:
    - Left child is at index 2*i + 1
    - Right child is at index 2*i + 2
    - Parent is at index (i-1) // 2
    
    This allows efficient heap operations without pointer manipulation.
    
    Attributes:
        heap_type: Type of heap (MIN or MAX)
        _heap_array: Internal array storing heap elements in level-order
    """
    
    def __init__(self, heap_type: HeapType = HeapType.MIN, initial_data: Optional[List[Any]] = None):
        """
        Initialize a heap.
        
        Args:
            heap_type: Type of heap (MIN or MAX), defaults to MIN
            initial_data: Optional list of initial values to heapify
            
        Example:
            >>> heap = Heap(HeapType.MIN)
            >>> heap = Heap(HeapType.MAX, [5, 3, 7, 1])
        """
        super().__init__()
        self.heap_type = heap_type
        self._heap_array: List[Any] = []
        
        if initial_data:
            self._heap_array = initial_data.copy()
            self._build_heap()
            self._sync_tree_from_array()
    
    def _sync_tree_from_array(self) -> None:
        """
        Synchronize the tree structure from the array representation.
        Rebuilds the BinaryTree structure from the heap array.
        
        This allows using inherited BinaryTree methods like print_tree, traversals, etc.
        """
        if not self._heap_array:
            self.root = None
            return
        
        # Create all nodes first
        nodes = [BinaryNode(val) for val in self._heap_array]
        
        # Set root
        self.root = nodes[0]
        
        # Connect parent-child relationships
        for i in range(len(nodes)):
            left_idx = self._left_child_index(i)
            right_idx = self._right_child_index(i)
            
            if left_idx < len(nodes):
                nodes[i].left = nodes[left_idx]
            
            if right_idx < len(nodes):
                nodes[i].right = nodes[right_idx]
    
    def _compare(self, a: Any, b: Any) -> bool:
        """
        Compare two values based on heap type.
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            True if a has higher priority than b based on heap type
        """
        if self.heap_type == HeapType.MIN:
            return a < b
        else:
            return a > b
    
    def _parent_index(self, index: int) -> int:
        """Get parent index of node at given index."""
        return (index - 1) // 2
    
    def _left_child_index(self, index: int) -> int:
        """Get left child index of node at given index."""
        return 2 * index + 1
    
    def _right_child_index(self, index: int) -> int:
        """Get right child index of node at given index."""
        return 2 * index + 2
    
    def _has_parent(self, index: int) -> bool:
        """Check if node at given index has a parent."""
        return self._parent_index(index) >= 0
    
    def _has_left_child(self, index: int) -> bool:
        """Check if node at given index has a left child."""
        return self._left_child_index(index) < len(self._heap_array)
    
    def _has_right_child(self, index: int) -> bool:
        """Check if node at given index has a right child."""
        return self._right_child_index(index) < len(self._heap_array)
    
    def _swap(self, index1: int, index2: int) -> None:
        """Swap elements at two indices in the heap array."""
        self._heap_array[index1], self._heap_array[index2] = self._heap_array[index2], self._heap_array[index1]
    
    def _heapify_up(self, index: int) -> None:
        """
        Move element at index up to maintain heap property.
        Used after insertion.
        
        Args:
            index: Index of element to heapify up
        """
        while self._has_parent(index):
            parent_idx = self._parent_index(index)
            if self._compare(self._heap_array[index], self._heap_array[parent_idx]):
                self._swap(index, parent_idx)
                index = parent_idx
            else:
                break
    
    def _heapify_down(self, index: int) -> None:
        """
        Move element at index down to maintain heap property.
        Used after extraction.
        
        The root node swaps its position along the path of the
        largest/smallest successors.
        
        Args:
            index: Index of element to heapify down
        """
        while self._has_left_child(index):
            # Find child with higher priority
            priority_child_idx = self._left_child_index(index)
            
            if self._has_right_child(index):
                right_idx = self._right_child_index(index)
                if self._compare(self._heap_array[right_idx], self._heap_array[priority_child_idx]):
                    priority_child_idx = right_idx
            
            # Check if we need to swap
            if self._compare(self._heap_array[priority_child_idx], self._heap_array[index]):
                self._swap(index, priority_child_idx)
                index = priority_child_idx
            else:
                break
    
    def _build_heap(self) -> None:
        """
        Build heap from unsorted data (heapify).
        Start from last non-leaf node and heapify down.
        
        Time complexity: O(n)
        """
        # Start from last non-leaf node
        start_idx = (len(self._heap_array) - 2) // 2
        for i in range(start_idx, -1, -1):
            self._heapify_down(i)
    
    def heap_insert(self, value: Any) -> None:
        """
        Insert a new value into the heap.
        
        Args:
            value: Value to insert
            
        Time complexity: O(log n)
        
        Example:
            >>> heap = Heap(HeapType.MIN)
            >>> heap.heap_insert(5)
            >>> heap.heap_insert(3)
            >>> heap.heap_insert(7)
        """
        self._heap_array.append(value)
        self._heapify_up(len(self._heap_array) - 1)
        self._sync_tree_from_array()
    
    def heap_extract(self) -> Any:
        """
        Extract and return the root element (min or max depending on heap type).
        
        The root node is removed from the heap.
        The leaf with the largest/smallest value moves to the root position.
        
        Returns:
            The root element (minimum for MIN heap, maximum for MAX heap)
            
        Raises:
            IndexError: If heap is empty
            
        Time complexity: O(log n)
        
        Example:
            >>> heap = Heap(HeapType.MIN, [5, 3, 7, 1])
            >>> heap.heap_extract()
            1
        """
        if self.is_heap_empty():
            raise IndexError("Cannot extract from empty heap")
        
        if len(self._heap_array) == 1:
            value = self._heap_array.pop()
            self._sync_tree_from_array()
            return value
        
        # Store root to return
        root = self._heap_array[0]
        
        # Move last element to root
        self._heap_array[0] = self._heap_array.pop()
        
        # Restore heap property
        self._heapify_down(0)
        self._sync_tree_from_array()
        
        return root
    
    def heap_peek(self) -> Any:
        """
        Return the root element without removing it.
        
        Returns:
            The root element (minimum for MIN heap, maximum for MAX heap)
            
        Raises:
            IndexError: If heap is empty
            
        Time complexity: O(1)
        """
        if self.is_heap_empty():
            raise IndexError("Cannot peek into empty heap")
        return self._heap_array[0]
    
    def is_heap_empty(self) -> bool:
        """
        Check if heap is empty.
        
        Returns:
            True if heap is empty, False otherwise
        """
        return len(self._heap_array) == 0
    
    def heap_size(self) -> int:
        """
        Get the number of elements in the heap.
        
        Returns:
            Number of elements
        """
        return len(self._heap_array)
    
    def heap_clear(self) -> None:
        """Clear all elements from the heap."""
        self._heap_array.clear()
        self.root = None
    
    def to_array(self) -> List[Any]:
        """
        Get a copy of the internal heap array.
        
        Note: This returns the heap in its internal representation,
        not in sorted order. Use heap_sort() for sorted output.
        
        Returns:
            Copy of heap data
        """
        return self._heap_array.copy()
    
    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return len(self._heap_array)
    
    def __bool__(self) -> bool:
        """Return True if heap is not empty."""
        return not self.is_heap_empty()
    
    def __str__(self) -> str:
        """String representation of the heap."""
        heap_type_str = "Min-Heap" if self.heap_type == HeapType.MIN else "Max-Heap"
        return f"{heap_type_str}({self._heap_array})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


def heap_sort(data: List[Any], reverse: bool = False) -> List[Any]:
    """
    Sort data using heap sort algorithm.
    
    The Heap Sort Algorithm:
    Prerequisite: n given valued nodes.
    1. The nodes are converted into a (Max-)Heap.
    2. The root node is removed from the heap and stored in an array/list.
    3. The leaf with the largest value moves to the root position.
    4. If the heap condition is not satisfied, the root node swaps its position 
       along the path of the largest successors.
    5. Go to step 2.
    
    Note: The heap sort algorithm is currently considered one of the most effective 
    sorting algorithms when dealing with many totally ordered data, especially when 
    duplicate data exists in the dataset.
    
    Args:
        data: List of data to sort
        reverse: If True, sort in descending order; if False, ascending order
        
    Returns:
        Sorted list
        
    Time complexity: O(n log n)
    Space complexity: O(n)
    
    Example:
        >>> heap_sort([5, 3, 7, 1, 9, 2])
        [1, 2, 3, 5, 7, 9]
        >>> heap_sort([5, 3, 7, 1, 9, 2], reverse=True)
        [9, 7, 5, 3, 2, 1]
    """
    if not data:
        return []
    
    # Step 1: Build heap
    # For ascending order, use MAX heap (extract max repeatedly gives descending, then reverse)
    # For descending order, use MIN heap (extract min repeatedly gives ascending, then reverse)
    heap_type = HeapType.MAX if not reverse else HeapType.MIN
    heap = Heap(heap_type, data)
    
    # Step 2-5: Extract all elements
    sorted_data = []
    while not heap.is_heap_empty():
        sorted_data.append(heap.heap_extract())
    
    # Reverse to get correct order
    if not reverse:
        sorted_data.reverse()
    else:
        sorted_data.reverse()
    
    return sorted_data


def heapify(data: List[Any], heap_type: HeapType = HeapType.MIN) -> Heap:
    """
    Convert a list into a heap in-place.
    
    Args:
        data: List to convert to heap
        heap_type: Type of heap to create
        
    Returns:
        Heap object containing the data
        
    Time complexity: O(n)
    
    Example:
        >>> heap = heapify([5, 3, 7, 1, 9, 2], HeapType.MIN)
        >>> heap.heap_peek()
        1
    """
    return Heap(heap_type, data)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("HEAP DATA STRUCTURE DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Min-Heap
    print("\n--- Example 1: Min-Heap ---")
    min_heap = Heap(HeapType.MIN)
    values = [10, 12, 13, 14, 11, 15, 16]
    print(f"Inserting values: {values}")
    for val in values:
        min_heap.heap_insert(val)
    
    print(f"Min-Heap: {min_heap}")
    print(f"Root (minimum): {min_heap.heap_peek()}")
    print("\nHeap tree structure (using inherited print_tree from BinaryTree):")
    min_heap.print_tree()
    
    # Example 2: Max-Heap
    print("\n--- Example 2: Max-Heap ---")
    max_heap = Heap(HeapType.MAX)
    values = [10, 12, 13, 14, 11, 15, 16]
    print(f"Inserting values: {values}")
    for val in values:
        max_heap.heap_insert(val)
    
    print(f"Max-Heap: {max_heap}")
    print(f"Root (maximum): {max_heap.heap_peek()}")
    print("\nHeap tree structure (using inherited print_tree from BinaryTree):")
    max_heap.print_tree()
    
    # Example 3: Extracting from Min-Heap
    print("\n--- Example 3: Extracting from Min-Heap ---")
    print("Extracting all elements in order:")
    extracted = []
    while not min_heap.is_heap_empty():
        extracted.append(min_heap.heap_extract())
    print(f"Extracted order: {extracted}")
    
    # Example 4: Heap Sort (Ascending)
    print("\n--- Example 4: Heap Sort (Ascending) ---")
    unsorted = [5, 3, 7, 1, 9, 2, 8, 4, 6]
    print(f"Unsorted: {unsorted}")
    sorted_asc = heap_sort(unsorted)
    print(f"Sorted (ascending): {sorted_asc}")
    
    # Example 5: Heap Sort (Descending)
    print("\n--- Example 5: Heap Sort (Descending) ---")
    unsorted = [5, 3, 7, 1, 9, 2, 8, 4, 6]
    print(f"Unsorted: {unsorted}")
    sorted_desc = heap_sort(unsorted, reverse=True)
    print(f"Sorted (descending): {sorted_desc}")
    
    # Example 6: Heapify existing data
    print("\n--- Example 6: Heapify existing data ---")
    data = [15, 10, 20, 8, 12, 25, 6]
    print(f"Original data: {data}")
    heap = heapify(data, HeapType.MIN)
    print(f"After heapify: {heap}")
    print("Heap tree structure:")
    heap.print_tree()
    
    # Example 7: Heap with duplicates
    print("\n--- Example 7: Heap Sort with duplicates ---")
    data_with_dupes = [5, 3, 5, 1, 3, 7, 1, 9, 3]
    print(f"Data with duplicates: {data_with_dupes}")
    sorted_dupes = heap_sort(data_with_dupes)
    print(f"Sorted: {sorted_dupes}")
    
    # Example 8: Heap with strings
    print("\n--- Example 8: Heap with strings ---")
    string_heap = Heap(HeapType.MIN)
    words = ["banana", "apple", "cherry", "date", "elderberry"]
    print(f"Inserting words: {words}")
    for word in words:
        string_heap.heap_insert(word)
    print(f"Min-Heap: {string_heap}")
    print(f"Root (alphabetically first): {string_heap.heap_peek()}")
    
    # Example 9: Heap properties and BinaryTree inheritance
    print("\n--- Example 9: Heap properties and BinaryTree inheritance ---")
    heap = Heap(HeapType.MIN, [10, 12, 13, 14, 11, 15, 16])
    print(f"Heap size: {heap.heap_size()}")
    print(f"Is empty: {heap.is_heap_empty()}")
    print(f"Internal array representation: {heap.to_array()}")
    print(f"\nInherited from BinaryTree:")
    print(f"- Node count: {heap.get_node_count()}")
    print(f"- Height: {heap.get_height()}")
    print(f"- Is complete: {heap.is_complete()}")
    print(f"- Is perfect: {heap.is_perfect()}")
    print(f"- Preorder traversal: {list(heap.traverse_preorder())}")
    print("\nData Ordering in Heaps:")
    print("- Data is 'partially' ordered (partial order)")
    print("- Compromise between completely unordered and completely ordered storage")
    print("- Fast insertion (O(log n)) and fast access to Min/Max (O(1))")
    
    print("\n" + "=" * 80)
