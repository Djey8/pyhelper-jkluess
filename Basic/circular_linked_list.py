# Create circular linked list
# Define Node class
class Node:
    def __init__(self, data):
        # Initialize a node with data and
        # an empty next pointer
        self.data = data   
        self.next = None     

# Define CircularLinkedList class
class CircularLinkedList:
    def __init__(self):
        # Initialize an empty CircularLinkedList
        # with a head pointer that points to None
        self.head = None
        
    # Method to add an element at the end
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            # If the list is empty
            # the new node should point to itself
            self.head = new_node
            self.head.next = self.head
        else:
            # If the list is not empty
            temp = self.head
            while temp.next != self.head:
                # traverse the list to the last node
                temp = temp.next
            # the last node should point to the new one
            temp.next = new_node        # append new element
            new_node.next = self.head   # last node points to head
            
    # Method to delete an element with a specific value
    def delete(self, key):
        if not self.head:
            print("List is empty.")
            return
        
        curr = self.head
        # Case: delete head
        if curr.data == key:
            if curr.next == self.head:
                # Only one element present
                # set head pointer to None
                self.head = None
                return
            else:
                # Multiple elements present
                # traverse the list to the last node
                temp = self.head
                while temp.next != self.head:
                    temp = temp.next
                # delete last node
                temp.next = self.head.next
                self.head = self.head.next
                return
        
        # Otherwise search and delete element in the interior
        prev = None
        while curr.next != self.head:
            prev = curr
            curr = curr.next
            if curr.data == key:
                prev.next = curr.next
                return
            
    # Method to output all elements
    def print_list(self):
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
        

# Example usage
if __name__ == "__main__":
    cll = CircularLinkedList()
    cll.append(1)
    cll.append(2)
    cll.append(3)
    cll.append('a')
    
    cll.print_list()
    
    print("\nNode with value 2 is deleted:")
    cll.delete(2) # Delete node with value 2
    cll.print_list()
    print("\nNode with value 5 is appended:")
    cll.append(5)
    cll.print_list() # Output list again
    
    