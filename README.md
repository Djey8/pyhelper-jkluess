# PyHelper

A collection of Python data structures for educational purposes at FOM.

## Features

This package includes implementations of:

### Basic Data Structures
- **Array** - Array implementations
- **Linked List** - Single linked list
- **Double Linked List** - Doubly linked list
- **Circular Linked List** - Circular linked list
- **Stack** - Stack data structure
- **List** - List implementations

### Complex Data Structures
- Coming soon...

## Installation

### Install from local directory (for development)

```bash
pip install -e .
```

### Install from local directory (regular installation)

```bash
pip install .
```

### Install from GitHub

```bash
pip install git+https://github.com/Djey8/PyHelper.git
```

## Usage

### Importable Classes (Ready to Use)

The following classes are designed to be imported and used in your projects:

#### LinkedList
```python
from pyhelper.basic.linked_list import LinkedList, Node

# Create an empty linked list
my_list = LinkedList()

# Create a linked list from existing data
my_list = LinkedList([1, 2, 3, 4, 5])

# Add elements
my_list.append(6)
my_list.append(7)

# Print the list
my_list.print_list()  # Output: 1 2 3 4 5 6 7

# Remove element at index
my_list.remove(2)  # Removes the element at index 2

# Get length
print(my_list.length)
```

#### DoubleLinkedList
```python
from pyhelper.basic.double_linked_list import DoubleLinkedList

# Create a doubly linked list
dll = DoubleLinkedList([10, 20, 30, 40])

# Add elements
dll.append(50)

# Print forward
dll.print_list(end=" -> ")

# Print backward
dll.print_list_backwards(end=" <- ")

# Remove element
dll.remove(1)  # Removes element at index 1
```

#### CircularLinkedList
```python
from pyhelper.basic.circular_linked_list import CircularLinkedList

# Create a circular linked list
cll = CircularLinkedList()

# Add elements
cll.append(1)
cll.append(2)
cll.append(3)

# Print the list
cll.print_list()  # Output: 1 -> 2 -> 3 -> (back to start)

# Delete by value
cll.delete(2)  # Removes the node with value 2
```

### Educational Examples (For Learning/Reference)

The following files contain educational examples showing different ways to work with data structures. **These are meant to be read and copied for learning purposes:**

#### Array Examples (`basic/array.py`)
- Working with NumPy arrays
- Array operations and manipulations
- Examples of 1D and 2D arrays

#### Stack Examples (`basic/stack.py`)
- Stack implementation using Python list
- Stack using `collections.deque`
- Stack using `queue.LifoQueue`

#### List Examples (`basic/list.py`)
- Basic Python list operations
- Reference for list methods

### Quick Import Reference
```python
# Import all importable classes
from pyhelper.basic.linked_list import LinkedList, Node
from pyhelper.basic.double_linked_list import DoubleLinkedList
from pyhelper.basic.circular_linked_list import CircularLinkedList
```

## Development

To install development dependencies:

```bash
pip install -e ".[dev]"
```

## License

MIT License - See LICENSE file for details

## Author

Jannis Kluess - FOM Student
