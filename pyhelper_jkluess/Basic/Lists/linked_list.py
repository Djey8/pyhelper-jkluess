""" 
Singly Linked List Data Structure

A singly linked list is a linear data structure where each element is a separate object
called a node. Each node contains data and a reference (link) to the next node in the sequence.
The first node is called the head, and the last node points to None.

This module provides a complete implementation with append, remove, and print operations.
"""


class Node:
    """
    A node in a singly linked list.
    
    Each node contains data and a reference to the next node in the sequence.
    
    Attributes:
        data: The data stored in this node (can be any type).
        next (Node): Reference to the next node, or None if this is the last node.
    """
    
    def __init__(self, data):
        """
        Initialize a new node.
        
        Args:
            data: The data to store in this node.
        """
        self.data = data
        self.next = None


class LinkedList:
    """
    A singly linked list implementation.
    
    Provides O(1) insertion at the end (if tail is tracked) and O(n) search/removal.
    Elements are stored in nodes that contain references to the next node.
    
    Attributes:
        Head (Node): The first node in the list, or None if empty.
        length (int): The number of nodes in the list.
    
    Example:
        >>> ll = LinkedList([1, 2, 3])
        >>> ll.append(4)
        >>> ll.length
        4
        >>> ll.remove(0)
        >>> ll.print_list()
        2 3 4
    """
    
    def __init__(self, data_list=None):
        """
        Initialize a new linked list.
        
        Args:
            data_list (list, optional): Initial data to populate the list.
                If provided, elements are appended in order.
        """
        self.Head = None
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
            O(n) where n is the number of elements, as we must traverse
            to the end of the list.
        
        Example:
            >>> ll = LinkedList()
            >>> ll.append(5)
            >>> ll.append(10)
            >>> ll.length
            2
        """
        new_node = Node(data)
        self.length += 1
        if not self.Head:
            self.Head = new_node
            return
        current = self.Head
        while current.next:
            current = current.next
        current.next = new_node
        
    def remove(self, index):
        """
        Remove the element at the specified index.
        
        Args:
            index (int): The zero-based index of the element to remove.
        
        Raises:
            IndexError: If index is negative or >= length.
        
        Time Complexity:
            O(n) where n is the index, as we must traverse to that position.
        
        Example:
            >>> ll = LinkedList([1, 2, 3])
            >>> ll.remove(1)
            >>> ll.print_list()
            1 3
        """
        if index < 0 or index >= self.length:
            raise IndexError("Index out of range")
        
        if index == 0:
            self.Head = self.Head.next
            self.length -= 1
            return
        
        current = self.Head
        for i in range(index - 1):
            current = current.next
        
        current.next = current.next.next
        self.length -= 1
    
    def print_list(self, end=" "):
        """
        Print all elements in the list.
        
        Args:
            end (str, optional): String to print between elements. Defaults to " ".
        
        Example:
            >>> ll = LinkedList([1, 2, 3])
            >>> ll.print_list(end=" -> ")
            1 -> 2 -> 3
        """
        node = self.Head
        while node is not None:
            if node.next is None:
                print(node.data)
            else:
                print(node.data, end=end)
            node = node.next
        

# Fill nodes and header of the linked list
my_list = LinkedList()
# 1st Node
node1 = Node(15)
# 2nd Node
node2 = Node(20)
# 3rd Node
node3 = Node(25)
# Set start node
my_list.Head = node1
# Set node pointers
node1.next = node2
node2.next = node3

my_list2 = LinkedList([99, 98, 97, 96])
my_list2.append(95)
# Output the list node by node:
print("List entries:")
print(my_list.Head.data, end=" ")
print(my_list.Head.next.data, end=" ")
print(my_list.Head.next.next.data, end=" ")
print("")
print("And once more the list entries with the while loop:")
my_list2.print_list(end=", ")
print("Length:", my_list2.length)

print("Delete the i-th element")
my_list2.remove(2)
my_list2.print_list()