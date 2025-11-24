import pytest
from pyhelper_jkluess.Basic.Lists.double_linked_list import DoubleLinkedList, Node


class TestDoubleNode:
    def test_node_creation(self):
        """Test creating a node with data"""
        node = Node(5)
        assert node.data == 5
        assert node.next is None
        assert node.prev is None


class TestDoubleLinkedList:
    def test_empty_list_creation(self):
        """Test creating an empty doubly linked list"""
        dll = DoubleLinkedList()
        assert dll.Head is None
        assert dll.Tail is None
        assert dll.length == 0
    
    def test_list_creation_with_data(self):
        """Test creating a doubly linked list with initial data"""
        dll = DoubleLinkedList([1, 2, 3])
        assert dll.length == 3
        assert dll.Head.data == 1
        assert dll.Tail.data == 3
    
    def test_append_to_empty_list(self):
        """Test appending to an empty list"""
        dll = DoubleLinkedList()
        dll.append(10)
        assert dll.Head.data == 10
        assert dll.Tail.data == 10
        assert dll.length == 1
        assert dll.Head.next is None
        assert dll.Head.prev is None
    
    def test_append_multiple_elements(self):
        """Test appending multiple elements"""
        dll = DoubleLinkedList()
        dll.append(1)
        dll.append(2)
        dll.append(3)
        assert dll.length == 3
        assert dll.Head.data == 1
        assert dll.Tail.data == 3
        assert dll.Head.next.data == 2
        assert dll.Tail.prev.data == 2
    
    def test_bidirectional_links(self):
        """Test that forward and backward links are correct"""
        dll = DoubleLinkedList([1, 2, 3])
        # Forward links
        assert dll.Head.next.data == 2
        assert dll.Head.next.next.data == 3
        # Backward links
        assert dll.Tail.prev.data == 2
        assert dll.Tail.prev.prev.data == 1
    
    def test_remove_head(self):
        """Test removing the head element"""
        dll = DoubleLinkedList([1, 2, 3])
        dll.remove(0)
        assert dll.length == 2
        assert dll.Head.data == 2
        assert dll.Head.prev is None
    
    def test_remove_tail(self):
        """Test removing the tail element"""
        dll = DoubleLinkedList([1, 2, 3])
        dll.remove(2)
        assert dll.length == 2
        assert dll.Tail.data == 2
        assert dll.Tail.next is None
    
    def test_remove_middle(self):
        """Test removing a middle element"""
        dll = DoubleLinkedList([1, 2, 3, 4])
        dll.remove(2)
        assert dll.length == 3
        assert dll.Head.next.next.data == 4
        assert dll.Tail.prev.data == 2
    
    def test_remove_single_element(self):
        """Test removing from a list with one element"""
        dll = DoubleLinkedList([1])
        dll.remove(0)
        assert dll.length == 0
        assert dll.Head is None
        assert dll.Tail is None
    
    def test_remove_invalid_index_negative(self):
        """Test removing with negative index raises error"""
        dll = DoubleLinkedList([1, 2, 3])
        with pytest.raises(IndexError):
            dll.remove(-1)
    
    def test_remove_invalid_index_too_large(self):
        """Test removing with index >= length raises error"""
        dll = DoubleLinkedList([1, 2, 3])
        with pytest.raises(IndexError):
            dll.remove(3)
    
    def test_print_list(self, capsys):
        """Test print_list method"""
        dll = DoubleLinkedList([1, 2, 3])
        dll.print_list()
        captured = capsys.readouterr()
        assert "1" in captured.out
        assert "2" in captured.out
        assert "3" in captured.out
    
    def test_print_list_backwards(self, capsys):
        """Test print_list_backwards method"""
        dll = DoubleLinkedList([1, 2, 3])
        dll.print_list_backwards()
        captured = capsys.readouterr()
        assert "3" in captured.out
        assert "2" in captured.out
        assert "1" in captured.out
    
    def test_mixed_data_types(self):
        """Test doubly linked list with mixed data types"""
        dll = DoubleLinkedList([1, "hello", 3.14, True])
        assert dll.length == 4
        assert dll.Head.data == 1
        assert dll.Tail.data == True
