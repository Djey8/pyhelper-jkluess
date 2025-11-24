# Linked Lists

**Node-based linear data structures** with three traversal patterns for different use cases.

## Core Concept

Linked lists store data in **nodes** connected by **pointers/references**:
- **Dynamic sizing**: No fixed capacity, grow/shrink as needed
- **Efficient insertions/deletions**: O(1) when you have the node reference
- **Trade-off**: O(n) access time (must traverse) vs O(1) array indexing

**Choose linked lists over arrays when**: Memory is limited, size changes frequently, or insertions/deletions are common.

## LinkedList - Single Direction

Basic linked list with forward-only traversal.

```python
from pyhelper_jkluess.Basic.Lists.linked_list import LinkedList

ll = LinkedList()
ll.append(10)
ll.append(20)
ll.append(30)
ll.print_list()  # 10 -> 20 -> 30 -> None

ll.remove(1)  # Remove index 1
ll.print_list()  # 10 -> 30 -> None
```

**When to use:** Memory-efficient, simple forward traversal only.

## DoubleLinkedList - Bidirectional

Links forward and backward between nodes.

```python
from pyhelper_jkluess.Basic.Lists.double_linked_list import DoubleLinkedList

dll = DoubleLinkedList()
dll.append(1)
dll.append(2)
dll.append(3)

dll.print_list()            # 1 <-> 2 <-> 3 <-> None
dll.print_list_backwards()  # 3 <-> 2 <-> 1 <-> None

dll.remove(1)
dll.print_list()            # 1 <-> 3 <-> None
```

**When to use:** Need backward navigation, LRU cache, undo/redo.

## CircularLinkedList - Loop Structure

Last node points back to first node.

```python
from pyhelper_jkluess.Basic.Lists.circular_linked_list import CircularLinkedList

cll = CircularLinkedList()
cll.append(1)
cll.append(2)
cll.append(3)
cll.print_list()  # 1 -> 2 -> 3 -> (back to 1)

cll.delete(2)
cll.print_list()  # 1 -> 3 -> (back to 1)
```

**When to use:** Round-robin scheduling, circular buffers.

## Performance

| Operation | All Types |
|-----------|-----------|
| Append | O(n) |
| Remove/Delete | O(n) |
| Search | O(n) |
