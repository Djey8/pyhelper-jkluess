import pytest
from pyhelper_jkluess.Basic.Lists.linked_list import LinkedList, Node


class TestNode:
    def test_node_creation(self):
        """Test creating a node with data"""
        node = Node(5)
        assert node.data == 5
        assert node.next is None
    
    def test_node_with_string(self):
        """Test node with string data"""
        node = Node("hello")
        assert node.data == "hello"
        assert node.next is None


class TestLinkedList:
    def test_empty_list_creation(self):
        """Test creating an empty linked list"""
        ll = LinkedList()
        assert ll.Head is None
        assert ll.length == 0
    
    def test_list_creation_with_data(self):
        """Test creating a linked list with initial data"""
        ll = LinkedList([1, 2, 3])
        assert ll.length == 3
        assert ll.Head.data == 1
        assert ll.Head.next.data == 2
        assert ll.Head.next.next.data == 3
    
    def test_append_to_empty_list(self):
        """Test appending to an empty list"""
        ll = LinkedList()
        ll.append(10)
        assert ll.Head.data == 10
        assert ll.length == 1
        assert ll.Head.next is None
    
    def test_append_multiple_elements(self):
        """Test appending multiple elements"""
        ll = LinkedList()
        ll.append(1)
        ll.append(2)
        ll.append(3)
        assert ll.length == 3
        assert ll.Head.data == 1
        assert ll.Head.next.data == 2
        assert ll.Head.next.next.data == 3
    
    def test_remove_head(self):
        """Test removing the head element"""
        ll = LinkedList([1, 2, 3])
        ll.remove(0)
        assert ll.length == 2
        assert ll.Head.data == 2
    
    def test_remove_middle(self):
        """Test removing a middle element"""
        ll = LinkedList([1, 2, 3, 4])
        ll.remove(2)
        assert ll.length == 3
        assert ll.Head.next.next.data == 4
    
    def test_remove_last(self):
        """Test removing the last element"""
        ll = LinkedList([1, 2, 3])
        ll.remove(2)
        assert ll.length == 2
        assert ll.Head.next.next is None
    
    def test_remove_invalid_index_negative(self):
        """Test removing with negative index raises error"""
        ll = LinkedList([1, 2, 3])
        with pytest.raises(IndexError):
            ll.remove(-1)
    
    def test_remove_invalid_index_too_large(self):
        """Test removing with index >= length raises error"""
        ll = LinkedList([1, 2, 3])
        with pytest.raises(IndexError):
            ll.remove(3)
    
    def test_remove_from_empty_list(self):
        """Test removing from empty list raises error"""
        ll = LinkedList()
        with pytest.raises(IndexError):
            ll.remove(0)
    
    def test_print_list(self, capsys):
        """Test print_list method"""
        ll = LinkedList([1, 2, 3])
        ll.print_list()
        captured = capsys.readouterr()
        assert "1" in captured.out
        assert "2" in captured.out
        assert "3" in captured.out
    
    def test_mixed_data_types(self):
        """Test linked list with mixed data types"""
        ll = LinkedList([1, "hello", 3.14, True])
        assert ll.length == 4
        assert ll.Head.data == 1
        assert ll.Head.next.data == "hello"
        assert ll.Head.next.next.data == 3.14
        assert ll.Head.next.next.next.data == True
