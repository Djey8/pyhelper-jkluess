# Basic Data Structures

Fundamental linked list implementations.

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
from Basic.Lists.linked_list import LinkedList

ll = LinkedList()
ll.append(10)
ll.append(20)
ll.print_list()  # 10 -> 20 -> None
```

For detailed documentation, see [Lists README](Lists/README.md).
