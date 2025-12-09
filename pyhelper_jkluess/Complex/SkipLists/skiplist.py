"""
Skip List Data Structure

A skip list is a probabilistic data structure that allows O(log n) search complexity
as well as O(log n) insertion within an ordered sequence of n elements. It uses a
hierarchy of linked lists that "skip" over elements to speed up search operations.

This module provides a complete implementation of a skip list with insert, search,
and delete operations.
"""

import random


class Node:
    """
    A node in the skip list.
    
    Each node contains a key-value pair and an array of forward pointers,
    one for each level the node participates in.
    
    Attributes:
        key: The key used for ordering elements in the skip list.
        value: The value associated with the key.
        forward (list): Array of forward pointers to next nodes at each level.
    """
    
    def __init__(self, key, value, level):
        """
        Initialize a skip list node.
        
        Args:
            key: The key for this node.
            value: The value associated with the key.
            level (int): The highest level this node participates in.
        """
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)


class SkipList:
    """
    A skip list implementation for efficient search, insert, and delete operations.
    
    Skip lists provide O(log n) average time complexity for search, insert, and
    delete operations through a probabilistic balancing scheme. The structure
    consists of multiple levels of linked lists, where higher levels "skip"
    over more elements.
    
    Attributes:
        max_level (int): Maximum number of levels in the skip list.
        header (Node): Sentinel header node that starts all levels.
        level (int): Current highest level with elements (0-indexed).
    
    Example:
        >>> sl = SkipList()
        >>> sl.insert(1, "one")
        >>> sl.insert(2, "two")
        >>> sl.search(1)
        'one'
        >>> sl.delete(1)
        >>> sl.search(1)
        None
    """
    
    def __init__(self, max_level=16):
        """
        Initialize an empty skip list.
        
        Args:
            max_level (int, optional): Maximum number of levels. Defaults to 16.
                A max_level of 16 can efficiently handle up to 2^16 (65536) elements.
        """
        self.max_level = max_level
        self.header = Node(None, None, max_level)
        self.level = 0
    
    def random_level(self):
        """
        Generate a random level for a new node.
        
        Uses a probabilistic approach where each level has a 50% chance of
        being included. This maintains the skip list's balanced properties
        on average.
        
        Returns:
            int: A random level between 0 and max_level (inclusive).
        
        Note:
            The probability of a node having level k is (1/2)^(k+1).
        """
        level = 0
        while random.random() < 0.5 and level < self.max_level:
            level += 1
        return level
    
    def search(self, key):
        """
        Search for a key in the skip list.
        
        Starts from the highest level and moves down, skipping elements that
        are too small, until finding the key or determining it doesn't exist.
        
        Args:
            key: The key to search for.
        
        Returns:
            The value associated with the key if found, None otherwise.
        
        Time Complexity:
            O(log n) on average
        
        Example:
            >>> sl = SkipList()
            >>> sl.insert(5, "five")
            >>> sl.search(5)
            'five'
            >>> sl.search(10)
            None
        """
        current = self.header
        
        # Start from highest level and move down
        for i in range(self.level, -1, -1):
            # Move forward while next node's key is less than search key
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        
        # Move to the lowest level (actual list)
        current = current.forward[0]
        
        # Check if we found the key
        if current and current.key == key:
            return current.value
        return None
    
    def insert(self, key, value):
        """
        Insert a key-value pair into the skip list.
        
        If the key already exists, updates its value. Otherwise, creates a new
        node with a random level and inserts it in the proper position.
        
        Args:
            key: The key to insert (must be comparable).
            value: The value to associate with the key.
        
        Time Complexity:
            O(log n) on average
        
        Example:
            >>> sl = SkipList()
            >>> sl.insert(3, "three")
            >>> sl.insert(1, "one")
            >>> sl.insert(3, "THREE")  # Updates existing key
            >>> sl.search(3)
            'THREE'
        """
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find position for insertion at each level
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # Update existing key
        if current and current.key == key:
            current.value = value
        else:
            # Insert new node
            new_level = self.random_level()
            
            # If new level is higher than current max, update header pointers
            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level
            
            # Create new node and insert at all levels
            new_node = Node(key, value, new_level)
            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node
    
    def delete(self, key):
        """
        Delete a key from the skip list.
        
        Removes the node with the specified key from all levels it participates in.
        If the key doesn't exist, the operation has no effect.
        
        Args:
            key: The key to delete.
        
        Returns:
            None
        
        Time Complexity:
            O(log n) on average
        
        Example:
            >>> sl = SkipList()
            >>> sl.insert(5, "five")
            >>> sl.delete(5)
            >>> sl.search(5)
            None
        """
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find the node to delete at each level
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # If key found, remove from all levels
        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]
    