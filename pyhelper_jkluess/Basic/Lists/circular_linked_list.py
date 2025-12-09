"""
Circular Linked List Data Structure

A circular linked list is a variation of a linked list where the last node points back
to the first node instead of None, forming a circle. This structure is useful for
applications that require cyclic iteration, such as round-robin scheduling or
implementing circular buffers.

This module provides a complete implementation with append, delete, and print operations.
"""


class Node:
    """
    A node in a circular linked list.
    
    Each node contains data and a reference to the next node. In a circular list,
    the last node's next pointer references the first node instead of None.
    
    Attributes:
        data: The data stored in this node (can be any type).
        next (Node): Reference to the next node in the circular sequence.
    """
    
    def __init__(self, data):
        """
        Initialize a new node.
        
        Args:
            data: The data to store in this node.
        """
        self.data = data   
        self.next = None


class CircularLinkedList:
    """
    A circular linked list implementation.
    
    In this structure, the last node points back to the first node, creating a circle.
    This allows infinite traversal and is useful for applications requiring cyclic access
    to elements.
    
    Attributes:
        head (Node): The entry point to the circular list, or None if empty.
    
    Example:
        >>> cll = CircularLinkedList()
        >>> cll.append(1)
        >>> cll.append(2)
        >>> cll.append(3)
        >>> cll.print_list()
        1 -> 2 -> 3 -> (back to start)
    """
    
    def __init__(self):
        """
        Initialize an empty circular linked list.
        """
        self.head = None
        
    def append(self, data):
        """
        Append a new element to the end of the circular list.
        
        For an empty list, the new node points to itself. For a non-empty list,
        the new node is inserted at the end and points back to the head.
        
        Args:
            data: The data to append to the list.
        
        Time Complexity:
            O(n) where n is the number of elements, as we must traverse
            to the end of the list.
        
        Example:
            >>> cll = CircularLinkedList()
            >>> cll.append(5)
            >>> cll.append(10)
        """
        new_node = Node(data)
        if not self.head:
            # If the list is empty, node points to itself
            self.head = new_node
            self.head.next = self.head
        else:
            # If the list is not empty, traverse to the last node
            temp = self.head
            while temp.next != self.head:
                temp = temp.next
            # Insert new node at the end
            temp.next = new_node
            new_node.next = self.head
            
    def delete(self, key):
        """
        Delete the first node with the specified value.
        
        Handles deletion of the head node, single-element lists, and interior nodes
        while maintaining the circular structure.
        
        Args:
            key: The value to search for and delete.
        
        Time Complexity:
            O(n) where n is the number of elements.
        
        Example:
            >>> cll = CircularLinkedList()
            >>> cll.append(1)
            >>> cll.append(2)
            >>> cll.append(3)
            >>> cll.delete(2)
            >>> cll.print_list()
            1 -> 3 -> (back to start)
        """
        if not self.head:
            print("List is empty.")
            return
        
        curr = self.head
        # Case: delete head node
        if curr.data == key:
            if curr.next == self.head:
                # Only one element present
                self.head = None
                return
            else:
                # Multiple elements present, find last node
                temp = self.head
                while temp.next != self.head:
                    temp = temp.next
                # Update last node to point to new head
                temp.next = self.head.next
                self.head = self.head.next
                return
        
        # Search and delete interior node
        prev = None
        while curr.next != self.head:
            prev = curr
            curr = curr.next
            if curr.data == key:
                prev.next = curr.next
                return
            
    def print_list(self):
        """
        Print all elements in the circular list.
        
        Traverses the entire circle once and indicates the circular nature
        by printing "(back to start)" at the end.
        
        Example:
            >>> cll = CircularLinkedList()
            >>> cll.append(1)
            >>> cll.append(2)
            >>> cll.print_list()
            1 -> 2 -> (back to start)
        """
        if not self.head:
            print("List is empty.")
            return
        temp = self.head
        while True:
            print(temp.data, end=" -> ")
            temp = temp.next
            if temp == self.head:
                break
        print("(back to start)")
    
    