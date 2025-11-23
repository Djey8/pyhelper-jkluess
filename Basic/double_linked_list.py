# Create new doubly linked list class
# Define Node class
# Syntax: self: self-defined reference

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoubleLinkedList:
    def __init__(self, data_list=None):
        self.Head = None
        self.Tail = None
        self.length = 0
        if data_list:
            for data in data_list:
                self.append(data)
    def append(self, data):
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
        node = self.Head
        while node is not None:
            if node.next is None:
                print(node.data)
            else:
                print(node.data, end=end)
            node = node.next
            
    def print_list_backwards(self, end=""):
        node = self.Tail
        while node is not None:
            if node.prev is None:
                print(node.data)
            else:
                print(node.data, end=end)
            node = node.prev
            
            
            
# Fill nodes and header of the doubly linked list
my_list = DoubleLinkedList()
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

my_list2 = DoubleLinkedList([99, 98, 97, 96])
my_list2.append(95)
# Output the list node by node:
print("List entries:")
print(my_list.Head.data, end=" ")
print(my_list.Head.next.data, end=" ")
print(my_list.Head.next.next.data, end=" ")
print("")
print("And once more the list entries with the while loop:")
my_list2.print_list(end=", ")
print("The list in reverse order:")
my_list2.print_list_backwards(end="; ")