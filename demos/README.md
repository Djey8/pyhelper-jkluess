# PyHelper Demos

This directory contains comprehensive, educational demonstration files for all data structures in the PyHelper package. Each demo file is a standalone tutorial covering all aspects of that data structure with practical examples and clear explanations.

## Available Demos

### Trees
- **`tree_demo.py`** - General Tree data structure (14 examples)
  - Manual creation, properties, traversals, generators
  - Import/export (adjacency matrix, adjacency list, nested structure)
  - Graph theory properties, visualization
  
- **`binary_tree_demo.py`** - Binary Tree specifics (10 examples)
  - Binary tree creation and traversals
  - Complete, perfect, and balanced trees
  - Expression trees with operators
  - BST sorting algorithm
  
- **`heap_demo.py`** - Heap data structure and heap sort (11 examples)
  - Min-heap and max-heap operations
  - Heap sort (ascending and descending)
  - Heapify operation
  - Different data types (numbers, strings)
  - Real-world use cases (priority queues, top-K)

### Graphs
- **`graph_demo.py`** - All graph types (12 examples)
  - Undirected/directed × weighted/unweighted
  - Real-world examples (social networks, tasks, cities, flights)
  - Traversals (DFS, BFS) and generators
  - Shortest path algorithms (BFS, Dijkstra)
  - Cycle detection, connectivity analysis
  - Minimum spanning trees (Kruskal, Prim)

### Lists
- **`linked_lists_demo.py`** - All linked list types (5 examples)
  - Singly linked list operations
  - Doubly linked list with bidirectional traversal
  - Circular linked list with round-robin use case
  - Comparison of all three types
  - Time complexity analysis

### Skip Lists
- **`skip_lists_demo.py`** - Probabilistic data structures (6 examples)
  - SkipList (deterministic) and ProbabilisticSkipList
  - Multi-level structure explanation
  - Search, insert, delete operations
  - Performance characteristics
  - Use cases (Redis, LevelDB)

## Running the Demos

Each demo file can be run independently:

```powershell
python demos/tree_demo.py
python demos/binary_tree_demo.py
python demos/heap_demo.py
python demos/graph_demo.py
python demos/linked_lists_demo.py
python demos/skip_lists_demo.py
```

## Learning Path

Suggested order for learning:

1. **Start with basics**: `linked_lists_demo.py` - Simple linear data structures
2. **Move to trees**: `tree_demo.py` → `binary_tree_demo.py` - Hierarchical structures
3. **Specialized trees**: `heap_demo.py` - Partial order and heap sort
4. **Skip lists**: `skip_lists_demo.py` - Alternative to balanced trees
5. **Graphs**: `graph_demo.py` - Most complex, builds on tree concepts

## Demo Structure

Each demo file follows a consistent structure:

- **Numbered examples** with clear docstrings explaining what's demonstrated
- **Real-world use cases** where applicable
- **Comparison sections** explaining trade-offs
- **Time complexity** analysis
- **Key takeaways** summary at the end
- **Visual aids** using emojis for easy navigation

## Additional Resources

- **examples/iterator_demo.py** - Demonstrates iterator patterns across data structures
- Source code documentation in each module
- Comprehensive test suite in `tests/` directory

## Features

All demos showcase:
- ✅ Creation and initialization
- ✅ Common operations (insert, delete, search)
- ✅ Traversal methods
- ✅ Import/export functionality
- ✅ Visualization where applicable
- ✅ Performance characteristics
- ✅ Real-world applications

## Questions or Issues?

If you have questions about any demo or data structure:
1. Check the docstrings in the source code
2. Review the test files in `tests/` for more usage examples
3. Consult the module documentation in the package

---

**Note**: All examples are self-contained and can be run independently. No external dependencies beyond the PyHelper package itself are required.
