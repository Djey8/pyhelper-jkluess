import pytest
from pyhelper_jkluess.Basic.Lists.circular_linked_list import CircularLinkedList, Node


class TestCircularNode:
    def test_node_creation(self):
        """Test creating a node with data"""
        node = Node(5)
        assert node.data == 5
        assert node.next is None


class TestCircularLinkedList:
    def test_empty_list_creation(self):
        """Test creating an empty circular linked list"""
        cll = CircularLinkedList()
        assert cll.head is None
    
    def test_append_to_empty_list(self):
        """Test appending to an empty list"""
        cll = CircularLinkedList()
        cll.append(10)
        assert cll.head.data == 10
        assert cll.head.next == cll.head  # Points to itself
    
    def test_append_multiple_elements(self):
        """Test appending multiple elements"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.append(2)
        cll.append(3)
        
        # Verify circular structure
        assert cll.head.data == 1
        assert cll.head.next.data == 2
        assert cll.head.next.next.data == 3
        assert cll.head.next.next.next == cll.head  # Last points back to head
    
    def test_circular_property(self):
        """Test that the list is truly circular"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.append(2)
        cll.append(3)
        
        # Traverse full circle
        current = cll.head
        count = 0
        while count < 10:  # Go around multiple times
            current = current.next
            count += 1
        # Should be able to traverse indefinitely
        assert current is not None
    
    def test_delete_from_empty_list(self, capsys):
        """Test deleting from empty list prints message"""
        cll = CircularLinkedList()
        cll.delete(1)
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()
    
    def test_delete_only_element(self):
        """Test deleting the only element in list"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.delete(1)
        assert cll.head is None
    
    def test_delete_head_with_multiple_elements(self):
        """Test deleting the head element when multiple elements exist"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.append(2)
        cll.append(3)
        cll.delete(1)
        
        assert cll.head.data == 2
        # Verify circular structure maintained
        assert cll.head.next.data == 3
        assert cll.head.next.next == cll.head
    
    def test_delete_middle_element(self):
        """Test deleting a middle element"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.append(2)
        cll.append(3)
        cll.delete(2)
        
        assert cll.head.data == 1
        assert cll.head.next.data == 3
        assert cll.head.next.next == cll.head
    
    def test_delete_last_element(self):
        """Test deleting the last element"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.append(2)
        cll.append(3)
        cll.delete(3)
        
        assert cll.head.data == 1
        assert cll.head.next.data == 2
        assert cll.head.next.next == cll.head
    
    def test_delete_nonexistent_element(self):
        """Test deleting an element that doesn't exist"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.append(2)
        cll.append(3)
        cll.delete(99)  # Should not crash
        
        # List should remain unchanged
        assert cll.head.data == 1
        assert cll.head.next.data == 2
        assert cll.head.next.next.data == 3
    
    def test_print_list_empty(self, capsys):
        """Test printing an empty list"""
        cll = CircularLinkedList()
        cll.print_list()
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()
    
    def test_print_list(self, capsys):
        """Test print_list method"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.append(2)
        cll.append(3)
        cll.print_list()
        captured = capsys.readouterr()
        assert "1" in captured.out
        assert "2" in captured.out
        assert "3" in captured.out
        assert "back to start" in captured.out.lower() or "anfang" in captured.out.lower()
    
    def test_mixed_data_types(self):
        """Test circular linked list with mixed data types"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.append("hello")
        cll.append(3.14)
        
        assert cll.head.data == 1
        assert cll.head.next.data == "hello"
        assert cll.head.next.next.data == 3.14
        assert cll.head.next.next.next == cll.head
    
    def test_append_after_delete_all(self):
        """Test appending after deleting all elements"""
        cll = CircularLinkedList()
        cll.append(1)
        cll.delete(1)
        cll.append(2)
        
        assert cll.head.data == 2
        assert cll.head.next == cll.head
