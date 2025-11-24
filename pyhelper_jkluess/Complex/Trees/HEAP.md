# Heap Data Structure

A **Heap** is a specialized binary tree data structure that satisfies the heap property. This implementation extends the `BinaryTree` class and provides both **Min-Heap** and **Max-Heap** variants with efficient array-based storage.

## Table of Contents
- [Definition](#definition)
- [Class Hierarchy](#class-hierarchy)
- [Heap Types](#heap-types)
- [Implementation Details](#implementation-details)
- [Heap Operations](#heap-operations)
- [Heap Sort Algorithm](#heap-sort-algorithm)
- [Usage Examples](#usage-examples)
- [Time Complexity](#time-complexity)
- [Testing](#testing)

## Definition

**Definition 5.10: Heap**

A heap is a binary, ordered, vertex-valued rooted tree with the following properties:

1. **Binary**: Every node has at most two children
2. **Complete**: All levels are fully filled except possibly the last level, which is filled from left to right
3. **Heap Property**:
   - **Min-Heap**: For every node, the parent's value is ≤ the values of its children
   - **Max-Heap**: For every node, the parent's value is ≥ the values of its children

**Key Characteristics**:
- The smallest (largest) value is always at the root (accessible in O(1) time)
- If h is the height of the heap, then: 2^h ≤ n ≤ 2^(h+1) - 1
- The root of the tree contains the minimum (Min-Heap) or maximum (Max-Heap) value
- Every subtree of a heap is also a heap

### Data Ordering in Heaps

In a heap, data is **"partially" ordered** (partial order):
- **Not completely unordered**: The heap property ensures parent-child relationships
- **Not completely ordered**: Siblings have no ordering relationship
- **Compromise**: Fast insertion O(log n) and fast access to Min/Max O(1)

## Class Hierarchy

```
Node (base class)
  └── BinaryNode (extends Node)

Tree (base class)
  └── BinaryTree (extends Tree)
      └── Heap (extends BinaryTree)
```

The `Heap` class inherits all functionality from `BinaryTree` and `Tree`, including:
- Tree visualization (`print_tree()`)
- Traversals (preorder, inorder, postorder, level-order)
- Tree properties (`get_height()`, `get_node_count()`, `is_complete()`, `is_perfect()`)
- Node relationships (`get_parent()`, `get_children()`, `get_ancestors()`, `get_descendants()`)

## Heap Types

```python
from pyhelper_jkluess.Complex.Trees.heap import HeapType

class HeapType(Enum):
    MIN = 1  # Min-Heap: smallest value at root
    MAX = 2  # Max-Heap: largest value at root
```

## Implementation Details

### Dual Representation

The heap maintains a **hybrid structure**:

1. **Array-based storage**: `_heap_array` stores values for efficient heap operations
2. **Tree structure**: Binary tree nodes for visualization and inherited operations

**Array Index Mapping**:
```
For node at index i:
- Parent node:  (i - 1) // 2
- Left child:   2 * i + 1
- Right child:  2 * i + 2
```

**Example**:
```
Array:  [10, 15, 20, 17, 25]
Index:   0   1   2   3   4

Tree representation:
       10
      /  \
    15    20
   /  \
  17  25

Nodes are numbered level by level, from left to right.
```

### Array-Tree Synchronization

The heap maintains synchronization between the array and tree representations:
- **`_heap_array`** is the source of truth for heap operations
- **`_sync_tree_from_array()`** rebuilds the tree structure when needed for visualization
- Tree structure is updated after insertion and extraction operations

## Heap Operations

### Core Operations

#### 1. Create Heap

```python
from pyhelper_jkluess.Complex.Trees.heap import Heap, HeapType

# Empty heap
min_heap = Heap(HeapType.MIN)
max_heap = Heap(HeapType.MAX)

# Heap from initial data (uses _build_heap)
min_heap = Heap(HeapType.MIN, [5, 3, 7, 1, 9])
```

#### 2. Insert Element

```python
heap.heap_insert(42)
# Time complexity: O(log n)
# Inserts value and heapifies up
```

**Heapify Up Process**:
1. Add element to end of array
2. Compare with parent
3. Swap if heap property violated
4. Repeat until heap property satisfied or root reached

#### 3. Extract Root (Min/Max)

```python
min_value = heap.heap_extract()
# Time complexity: O(log n)
# Removes and returns root value
```

**Heapify Down Process**:
1. Remove root and save value
2. Move last element to root
3. Compare with children
4. Swap with smallest/largest child if heap property violated
5. Repeat until heap property satisfied or leaf reached

#### 4. Peek Root

```python
min_value = heap.heap_peek()
# Time complexity: O(1)
# Returns root value without removing
```

#### 5. Other Operations

```python
size = heap.heap_size()           # Get number of elements
is_empty = heap.is_heap_empty()   # Check if empty
heap.heap_clear()                 # Remove all elements
array = heap.to_array()           # Convert to array
```

### Inherited Operations

Since `Heap` extends `BinaryTree`, you can use all tree operations:

```python
# Visualization
heap.print_tree()

# Tree properties
height = heap.get_height()
count = heap.get_node_count()
is_complete = heap.is_complete()  # Always True for valid heap
is_perfect = heap.is_perfect()

# Traversals
preorder = list(heap.traverse_preorder())
inorder = list(heap.traverse_inorder())
postorder = list(heap.traverse_postorder())
levelorder = list(heap.traverse_level_order())
```

## Heap Sort Algorithm

### Algorithm Description

**The Heap Sort Algorithm**:

Prerequisite: n given valued nodes.

1. The nodes are converted into a (Max-)Heap.
2. The root node is removed from the heap and stored in an array/list.
3. The leaf with the largest value moves to the root position.
4. If the heap condition is not satisfied, the root node swaps its position along the path of the largest successors.
5. Go to step 2.

**Note**: The heap sort algorithm is currently considered one of the most effective sorting algorithms when dealing with many totally ordered data, especially when duplicate data exists in the dataset.

### Implementation

```python
from pyhelper_jkluess.Complex.Trees.heap import heap_sort

# Ascending order (using Max-Heap)
sorted_asc = heap_sort([5, 3, 7, 1, 9])  
# Output: [1, 3, 5, 7, 9]

# Descending order (using Min-Heap)
sorted_desc = heap_sort([5, 3, 7, 1, 9], reverse=True)
# Output: [9, 7, 5, 3, 1]
```

**Time Complexity**: O(n log n)
**Space Complexity**: O(n)

### Heapify Function

Convert a list to a heap in O(n) time:

```python
from pyhelper_jkluess.Complex.Trees.heap import heapify, HeapType

heap = heapify([5, 3, 7, 1, 9], HeapType.MIN)
# Returns a Heap object
```

## Usage Examples

### Example 1: Min-Heap Operations

```python
from pyhelper_jkluess.Complex.Trees.heap import Heap, HeapType

# Create min-heap
heap = Heap(HeapType.MIN)

# Insert elements
for value in [5, 3, 7, 1, 9, 4, 6]:
    heap.heap_insert(value)

# Peek minimum
print(heap.heap_peek())  # Output: 1

# Extract minimum values
print(heap.heap_extract())  # Output: 1
print(heap.heap_extract())  # Output: 3
print(heap.heap_extract())  # Output: 4

# Visualize heap
heap.print_tree()
# Output:
#     5
#    / \
#   9   6
#  /
# 7
```

### Example 2: Max-Heap Operations

```python
from pyhelper_jkluess.Complex.Trees.heap import Heap, HeapType

# Create max-heap with initial data
heap = Heap(HeapType.MAX, [5, 3, 7, 1, 9, 4, 6])

# Peek maximum
print(heap.heap_peek())  # Output: 9

# Extract maximum values
print(heap.heap_extract())  # Output: 9
print(heap.heap_extract())  # Output: 7
print(heap.heap_extract())  # Output: 6

# Check properties
print(f"Height: {heap.get_height()}")
print(f"Is complete: {heap.is_complete()}")
```

### Example 3: Heap Sort

```python
from pyhelper_jkluess.Complex.Trees.heap import heap_sort

# Sort in ascending order
data = [64, 34, 25, 12, 22, 11, 90]
sorted_data = heap_sort(data)
print(sorted_data)  # Output: [11, 12, 22, 25, 34, 64, 90]

# Sort in descending order
sorted_desc = heap_sort(data, reverse=True)
print(sorted_desc)  # Output: [90, 64, 34, 25, 22, 12, 11]
```

### Example 4: Using Inherited Tree Methods

```python
from pyhelper_jkluess.Complex.Trees.heap import Heap, HeapType

heap = Heap(HeapType.MIN, [10, 11, 12, 13, 14, 15, 16])

# Tree properties
print(f"Node count: {heap.get_node_count()}")
print(f"Height: {heap.get_height()}")
print(f"Is complete: {heap.is_complete()}")
print(f"Is perfect: {heap.is_perfect()}")

# Traversals
print(f"Preorder: {list(heap.traverse_preorder())}")
print(f"Inorder: {list(heap.traverse_inorder())}")
print(f"Level-order: {list(heap.traverse_level_order())}")

# Visualization
heap.print_tree()
```

### Example 5: Different Data Types

```python
from pyhelper_jkluess.Complex.Trees.heap import Heap, HeapType

# Float heap
float_heap = Heap(HeapType.MIN, [3.14, 2.71, 1.41, 1.73])
print(float_heap.heap_extract())  # Output: 1.41

# String heap
string_heap = Heap(HeapType.MIN, ["zebra", "apple", "mango", "banana"])
print(string_heap.heap_extract())  # Output: "apple"

# Negative numbers
neg_heap = Heap(HeapType.MAX, [-5, -1, -10, -3])
print(neg_heap.heap_extract())  # Output: -1
```

## Time Complexity

| Operation | Time Complexity | Description |
|-----------|----------------|-------------|
| `heap_insert(value)` | O(log n) | Insert and heapify up |
| `heap_extract()` | O(log n) | Remove root and heapify down |
| `heap_peek()` | O(1) | View root value |
| `heap_size()` | O(1) | Get number of elements |
| `is_heap_empty()` | O(1) | Check if empty |
| `heap_clear()` | O(1) | Remove all elements |
| `to_array()` | O(1) | Get array copy |
| `_build_heap()` | O(n) | Build heap from unsorted data |
| `_heapify_up(i)` | O(log n) | Restore heap property upward |
| `_heapify_down(i)` | O(log n) | Restore heap property downward |
| `heap_sort(data)` | O(n log n) | Sort using heap |

**Inherited from BinaryTree** (all operate on tree structure):
- `print_tree()`: O(n)
- `traverse_*()`: O(n)
- `get_height()`: O(n)
- `get_node_count()`: O(n)
- `is_complete()`: O(n)
- `is_perfect()`: O(n)

## Testing

The heap implementation has comprehensive test coverage with 38 tests across multiple categories:

### Test Categories

1. **TestMinHeap** (8 tests): Basic min-heap operations
2. **TestMaxHeap** (3 tests): Basic max-heap operations
3. **TestHeapOperations** (5 tests): General operations (clear, to_array, operators)
4. **TestHeapSort** (9 tests): Heap sort algorithm variations
5. **TestHeapify** (3 tests): Heapify function
6. **TestHeapWithDifferentTypes** (3 tests): Floats, strings, negative numbers
7. **TestHeapProperties** (3 tests): Parent-child relationships, heap property
8. **TestEdgeCases** (4 tests): Single element, two elements, all equal, large heap (1000 elements)

### Run Tests

```bash
# Run all heap tests
pytest tests/test_heap.py -v

# Run specific test class
pytest tests/test_heap.py::TestMinHeap -v

# Run specific test
pytest tests/test_heap.py::TestMinHeap::test_insert_and_peek -v
```

### Example Test Output

```
tests/test_heap.py::TestMinHeap::test_empty_heap PASSED
tests/test_heap.py::TestMinHeap::test_insert_and_peek PASSED
tests/test_heap.py::TestMinHeap::test_extract_min PASSED
tests/test_heap.py::TestMinHeap::test_multiple_operations PASSED
tests/test_heap.py::TestHeapSort::test_heap_sort_ascending PASSED
tests/test_heap.py::TestHeapSort::test_heap_sort_descending PASSED
...
================================ 38 passed in 1.52s ================================
```

## Performance Characteristics

**Strengths**:
- ✅ O(1) access to minimum/maximum value
- ✅ O(log n) insertion and extraction
- ✅ O(n) heap construction from unsorted data
- ✅ O(n log n) heap sort (in-place variant possible)
- ✅ Guaranteed complete tree structure (space efficient)
- ✅ Better worst-case than quicksort

**Use Cases**:
- Priority queues
- Heap sort algorithm
- Finding k smallest/largest elements
- Median maintenance (with two heaps)
- Event-driven simulation
- Dijkstra's shortest path algorithm
- Huffman coding

## Comparison with Other Data Structures

| Operation | Heap | Binary Search Tree | Sorted Array |
|-----------|------|-------------------|--------------|
| Find Min/Max | O(1) | O(log n) - O(n) | O(1) |
| Insert | O(log n) | O(log n) - O(n) | O(n) |
| Extract Min/Max | O(log n) | O(log n) - O(n) | O(n) |
| Search arbitrary | O(n) | O(log n) - O(n) | O(log n) |
| Space | O(n) | O(n) | O(n) |
| In-order traversal | O(n log n) | O(n) | O(n) |

## Implementation Notes

### Design Decisions

1. **Hybrid Structure**: Maintains both array and tree representations
   - Array for efficient heap operations
   - Tree for visualization and inherited functionality

2. **Lazy Tree Synchronization**: Tree structure only updated when needed
   - After insertion/extraction
   - On demand for visualization/traversal

3. **Method Naming**: All heap-specific methods prefixed with `heap_`
   - Avoids naming conflicts with inherited methods
   - Clear distinction between heap operations and tree operations

4. **Type Flexibility**: Supports any comparable data type
   - Numbers (int, float)
   - Strings
   - Custom objects with `__lt__`, `__le__`, `__gt__`, `__ge__` methods

5. **Error Handling**: Raises appropriate exceptions
   - `IndexError` for operations on empty heap
   - `TypeError` for uncomparable types

## Related Documentation

- [BinaryTree Documentation](BINARY_TREE.md)
- [Tree Documentation](README.md)
- [Heap Sort Algorithm Details](BINARY_TREE.md#tree-sorting-algorithm)

## References

- **Definition 5.10**: Formal heap definition from data structures theory
- **Heap Property**: Partial ordering constraint for parent-child relationships
- **Array Representation**: Complete binary tree mapping to array indices
- **Heap Sort**: O(n log n) comparison-based sorting algorithm
