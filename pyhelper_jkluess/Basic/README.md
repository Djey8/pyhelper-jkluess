# Basic Data Structures

**Linear data structures** - the foundation of computer science data organization.

## Core Concept: Linked Lists

Unlike arrays, linked lists use **nodes with pointers** for dynamic, memory-efficient storage:
- No pre-allocated size needed (grow/shrink dynamically)
- Efficient insertions/deletions (no shifting elements)
- Trade-off: O(n) access time vs O(1) for arrays

## Available Modules

### Lists (Production-Ready)
- **LinkedList** - Forward-only traversal
- **DoubleLinkedList** - Bidirectional traversal
- **CircularLinkedList** - Circular structure

See [Lists README](Lists/README.md) for usage examples.

### Educational Files
- `array.py` and `stack.py` - Learning reference only, not for production use.

## Quick Example

```python
from pyhelper_jkluess.Basic.Lists.linked_list import LinkedList

ll = LinkedList()
ll.append(10)
ll.append(20)
ll.print_list()  # 10 -> 20 -> None
```

For detailed documentation, see [Lists README](Lists/README.md).
