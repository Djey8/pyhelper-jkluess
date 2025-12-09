"""
Probabilistic Skip List Data Structure

A probabilistic skip list is a randomized data structure that provides expected O(log n)
time complexity for search, insertion, and deletion operations. Unlike balanced trees that
require complex rebalancing, skip lists achieve balance through randomization, where each
node has a randomly determined height that determines how many levels it participates in.

This implementation uses a 50% probability for height generation, making the expected
structure similar to a balanced binary tree but with simpler implementation.

The skip list consists of multiple levels of linked lists, where:
- Level 0 contains all elements
- Higher levels contain a subset of elements for faster traversal
- A sentinel node with value -infinity starts all levels
"""

from random import choice


class Node:
    """
    A node in the probabilistic skip list.
    
    Each node has a randomly determined height and maintains forward pointers
    at each level it participates in, allowing multi-level skipping during traversal.
    
    Attributes:
        data: The value stored in this node (must be comparable).
        next_nodes (list): Array of forward pointers, one for each level.
        height (int): The number of levels this node participates in.
    """
    
    def __init__(self, data, height):
        """
        Initialize a skip list node.
        
        Args:
            data: The value to store in this node.
            height (int): The number of levels for this node (1 to max_height).
        """
        self.data = data
        self.next_nodes = [None] * height
        self.height = height


class ProbabilisticSkipList:
    """
    A probabilistic skip list implementation.
    
    Uses randomization to maintain balance without complex rebalancing operations.
    Each node's height is determined probabilistically, with approximately 50% of nodes
    at each successive level, creating an expected O(log n) structure.
    
    Attributes:
        max_height (int): Maximum number of levels allowed in the skip list.
        sentinel (Node): A sentinel node with value -infinity that starts all levels.
        current_height (int): The current highest level with actual data nodes.
    
    Example:
        >>> psl = ProbabilisticSkipList()
        >>> psl.add(3)
        >>> psl.add(1)
        >>> psl.add(5)
        >>> psl.find(3)
        3
        >>> psl.display()
        [1, 3, 5]
        >>> psl.remove(3)
        True
        >>> psl.display()
        [1, 5]
    """
    
    def __init__(self, max_height=10):
        """
        Initialize an empty probabilistic skip list.
        
        Args:
            max_height (int, optional): Maximum number of levels. Defaults to 10.
                A max_height of 10 can efficiently handle up to 2^10 (1024) elements.
        """
        self.max_height = max_height
        self.sentinel = Node(float('-inf'), max_height)
        self.current_height = 1
        
    def _generate_height(self):
        """
        Generate a random height for a new node.
        
        Uses a coin-flip approach where each level has a 50% probability of being
        included. This maintains the probabilistic balance of the skip list.
        
        Returns:
            int: A random height between 1 and max_height (inclusive).
        
        Note:
            The probability of height h is (1/2)^h, ensuring that approximately
            half the nodes at level i also appear at level i+1.
        """
        height = 1
        while height < self.max_height and choice([True, False]):
            height += 1
        return height
    
    def find(self, target):
        """
        Search for a value in the skip list.
        
        Starts from the highest level and descends, skipping over elements that
        are too small. This multi-level approach provides logarithmic search time.
        
        Args:
            target: The value to search for (must be comparable).
        
        Returns:
            The value if found, None otherwise.
        
        Time Complexity:
            O(log n) expected time
        
        Example:
            >>> psl = ProbabilisticSkipList()
            >>> psl.add(5)
            >>> psl.add(10)
            >>> psl.find(5)
            5
            >>> psl.find(7)
            None
        """
        node = self.sentinel
        
        # Traverse from highest level down
        for level in range(self.current_height - 1, -1, -1):
            # Move forward while next node is less than target
            while (node.next_nodes[level] and 
                   node.next_nodes[level].data < target):
                node = node.next_nodes[level]
        
        # Check the candidate node at level 0
        candidate = node.next_nodes[0]
        return candidate.data if candidate and candidate.data == target else None
    
    def add(self, value):
        """
        Add a value to the skip list.
        
        Generates a random height for the new node and inserts it at all appropriate
        levels, maintaining sorted order at each level.
        
        Args:
            value: The value to add (must be comparable).
        
        Time Complexity:
            O(log n) expected time
        
        Example:
            >>> psl = ProbabilisticSkipList()
            >>> psl.add(3)
            >>> psl.add(1)
            >>> psl.add(5)
            >>> psl.display()
            [1, 3, 5]
        """
        predecessors = [None] * self.max_height
        node = self.sentinel
        
        # Find insertion point at each level
        for level in range(self.current_height - 1, -1, -1):
            while (node.next_nodes[level] and 
                   node.next_nodes[level].data < value):
                node = node.next_nodes[level]
            predecessors[level] = node
        
        # Generate random height for new node
        new_height = self._generate_height()
        
        # Update current height if necessary
        if new_height > self.current_height:
            for level in range(self.current_height, new_height):
                predecessors[level] = self.sentinel
            self.current_height = new_height
        
        # Create and insert new node at all levels
        new_node = Node(value, new_height)
        
        for level in range(new_height):
            new_node.next_nodes[level] = predecessors[level].next_nodes[level]
            predecessors[level].next_nodes[level] = new_node
    
    def remove(self, target):
        """
        Remove a value from the skip list.
        
        Searches for the target value and removes it from all levels it participates in,
        maintaining the skip list structure.
        
        Args:
            target: The value to remove.
        
        Returns:
            bool: True if the value was found and removed, False otherwise.
        
        Time Complexity:
            O(log n) expected time
        
        Example:
            >>> psl = ProbabilisticSkipList()
            >>> psl.add(1)
            >>> psl.add(2)
            >>> psl.add(3)
            >>> psl.remove(2)
            True
            >>> psl.display()
            [1, 3]
            >>> psl.remove(5)
            False
        """
        predecessors = [None] * self.max_height
        node = self.sentinel
        
        # Find node to delete at each level
        for level in range(self.current_height - 1, -1, -1):
            while (node.next_nodes[level] and 
                   node.next_nodes[level].data < target):
                node = node.next_nodes[level]
            predecessors[level] = node
        
        target_node = node.next_nodes[0]
        
        # Remove node from all levels if found
        if target_node and target_node.data == target:
            for level in range(target_node.height):
                predecessors[level].next_nodes[level] = target_node.next_nodes[level]
            return True
        return False
    
    def display(self):
        """
        Return all values in the skip list as a sorted list.
        
        Traverses the bottom level (which contains all elements) and collects
        all values in order.
        
        Returns:
            list: All values in the skip list in sorted order.
        
        Time Complexity:
            O(n) where n is the number of elements.
        
        Example:
            >>> psl = ProbabilisticSkipList()
            >>> psl.add(5)
            >>> psl.add(1)
            >>> psl.add(3)
            >>> psl.display()
            [1, 3, 5]
        """
        result = []
        node = self.sentinel.next_nodes[0]
        while node:
            result.append(node.data)
            node = node.next_nodes[0]
        return result