"""
Doubly Linked List Data Structure

A doubly linked list is a linear data structure where each node contains references
to both the next and previous nodes. This allows efficient traversal in both directions
and O(1) insertion/deletion at both ends when head and tail pointers are maintained.

This module provides a complete implementation with bidirectional traversal support.
"""


class Node:
    """
    A node in a doubly linked list.
    
    Each node contains data and references to both the next and previous nodes,
    enabling bidirectional traversal.
    
    Attributes:
        data: The data stored in this node (can be any type).
        next (Node): Reference to the next node, or None if this is the last node.
        prev (Node): Reference to the previous node, or None if this is the first node.
    """
    
    def __init__(self, data):
        """
        Initialize a new node.
        
        Args:
            data: The data to store in this node.
        """
        self.data = data
        self.next = None
        self.prev = None


class DoubleLinkedList:
    """
    A doubly linked list implementation.
    
    Maintains both head and tail pointers for efficient operations at both ends.
    Each node has references to both next and previous nodes, allowing bidirectional
    traversal and more efficient insertions/deletions compared to singly linked lists.
    
    Attributes:
        Head (Node): The first node in the list, or None if empty.
        Tail (Node): The last node in the list, or None if empty.
        length (int): The number of nodes in the list.
    
    Example:
        >>> dll = DoubleLinkedList([1, 2, 3])
        >>> dll.append(4)
        >>> dll.print_list(end=\" -> \")
        1 -> 2 -> 3 -> 4
        >>> dll.print_list_backwards(end=\" <- \")
        4 <- 3 <- 2 <- 1
    """
    
    def __init__(self, data_list=None):
        """
        Initialize a new doubly linked list.
        
        Args:
            data_list (list, optional): Initial data to populate the list.
                If provided, elements are appended in order.
        """
        self.Head = None
        self.Tail = None
        self.length = 0
        if data_list:
            for data in data_list:
                self.append(data)
    
    def append(self, data):
        """
        Append a new element to the end of the list.
        
        Args:
            data: The data to append to the list.
        
        Time Complexity:
            O(1) due to maintaining a tail pointer.
        
        Example:
            >>> dll = DoubleLinkedList()
            >>> dll.append(5)
            >>> dll.append(10)
            >>> dll.length
            2
        """
        new_node = Node(data)
        self.length += 1
        if not self.Head:
            self.Head = new_node
            self.Tail = new_node
            return
        self.Tail.next = new_node
        new_node.prev = self.Tail
        self.Tail = new_node
        
    def remove(self, index):
        """
        Remove the element at the specified index.
        
        Handles removal of head, tail, and middle elements efficiently by
        updating both next and prev pointers.
        
        Args:
            index (int): The zero-based index of the element to remove.
        
        Raises:
            IndexError: If index is negative or >= length.
        
        Time Complexity:
            O(n) in the worst case, but O(1) for head and tail removal.
        
        Example:
            >>> dll = DoubleLinkedList([1, 2, 3, 4])
            >>> dll.remove(2)
            >>> dll.print_list()
            1 2 4
        """
        if index < 0 or index >= self.length:
            raise IndexError("Index out of range")
        
        if index == 0:
            if self.Head.next:
                self.Head.next.prev = None
            self.Head = self.Head.next
            if self.length == 1:
                self.Tail = None
            self.length -= 1
            return
        
        if index == self.length - 1:
            self.Tail.prev.next = None
            self.Tail = self.Tail.prev
            self.length -= 1
            return
        
        current = self.Head
        for i in range(index):
            current = current.next
        
        current.prev.next = current.next
        current.next.prev = current.prev
        self.length -= 1
    
    def print_list(self, end=""):
        """
        Print all elements in the list from head to tail.
        
        Args:
            end (str, optional): String to print between elements. Defaults to "".
        
        Example:
            >>> dll = DoubleLinkedList([1, 2, 3])
            >>> dll.print_list(end=\" -> \")
            1 -> 2 -> 3
        """
        node = self.Head
        while node is not None:
            if node.next is None:
                print(node.data)
            else:
                print(node.data, end=end)
            node = node.next
            
    def print_list_backwards(self, end=""):
        """
        Print all elements in the list from tail to head.
        
        Demonstrates the advantage of doubly linked lists by traversing backwards
        using the prev pointers.
        
        Args:
            end (str, optional): String to print between elements. Defaults to "".
        
        Example:
            >>> dll = DoubleLinkedList([1, 2, 3])
            >>> dll.print_list_backwards(end=\" <- \")
            3 <- 2 <- 1
        """
        node = self.Tail
        while node is not None:
            if node.prev is None:
                print(node.data)
            else:
                print(node.data, end=end)
            node = node.prev