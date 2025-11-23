# Skip Lists

Probabilistic data structures for fast sorted operations (O(log n)).

## SkipList - Key-Value Store

Dictionary with O(log n) operations. Keys stay sorted.

```python
from Complex.SkipLists.skiplist import SkipList

sl = SkipList(max_level=4)

# Insert key-value pairs
sl.insert(10, "ten")
sl.insert(20, "twenty")
sl.insert(5, "five")

# Search by key
print(sl.search(10))  # "ten"
print(sl.search(7))   # None

# Update existing key
sl.insert(10, "TEN")
print(sl.search(10))  # "TEN"

# Delete
sl.delete(20)
```

**Use for:** Dictionaries, database indexes, caches with key lookup.

## ProbabilisticSkipList - Sorted Set

Stores sorted values with probabilistic balancing.

```python
from Complex.SkipLists.probabilisticskiplist import ProbabilisticSkipList

psl = ProbabilisticSkipList(max_height=10)

# Add values (auto-sorted)
psl.add(30)
psl.add(10)
psl.add(20)

# Find
print(psl.find(20))   # 20
print(psl.find(100))  # None

# Remove
psl.remove(10)

# Display structure
psl.display()
```

**Use for:** Sorted sets, priority queues, range queries.

## How It Works

Skip lists use multiple levels for fast traversal:

```
Level 2: 5 ---------------> 20 --> NULL
Level 1: 5 -----> 10 ----> 20 --> NULL
Level 0: 5 -> 10 -> 15 -> 20 --> NULL
```

Higher levels = express lanes for faster search.

## Key Differences

| Feature | SkipList | ProbabilisticSkipList |
|---------|----------|----------------------|
| Stores | Key-value pairs | Values only |
| Insert | `insert(key, value)` | `add(value)` |
| Search | `search(key)` | `find(value)` |
| Delete | `delete(key)` | `remove(value)` |
| Use | Dictionary | Sorted set |

## Performance

All operations: **O(log n)** average time
