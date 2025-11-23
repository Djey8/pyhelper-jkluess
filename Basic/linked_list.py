# Create new linked list class
# Define Node class
# Syntax: self: self-defined reference

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self, data_list=None):
        self.Head = None
        self.length = 0
        if data_list:
            for data in data_list:
                self.append(data)
    
    def append(self, data):
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