"""
Tests for Heap data structure implementation
"""

import pytest
from pyhelper_jkluess.Complex.Trees.heap import Heap, HeapType, heap_sort, heapify


class TestMinHeap:
    """Tests for Min-Heap functionality"""
    
    def test_empty_heap(self):
        """Test empty heap properties"""
        heap = Heap(HeapType.MIN)
        assert heap.is_heap_empty()
        assert heap.heap_size() == 0
        assert len(heap) == 0
        assert not heap  # __bool__ should return False
        
    def test_insert_single_element(self):
        """Test inserting single element"""
        heap = Heap(HeapType.MIN)
        heap.heap_insert(5)
        assert not heap.is_heap_empty()
        assert heap.heap_size() == 1
        assert heap.heap_peek() == 5
        
    def test_insert_multiple_elements(self):
        """Test inserting multiple elements"""
        heap = Heap(HeapType.MIN)
        values = [10, 12, 13, 14, 11, 15, 16]
        for val in values:
            heap.heap_insert(val)
        
        assert heap.heap_size() == 7
        assert heap.heap_peek() == 10  # Minimum should be at root
        
    def test_extract_min(self):
        """Test extracting minimum element"""
        heap = Heap(HeapType.MIN, [10, 12, 13, 14, 11, 15, 16])
        
        # Extract in order - should get sorted sequence
        extracted = []
        while not heap.is_heap_empty():
            extracted.append(heap.heap_extract())
        
        assert extracted == [10, 11, 12, 13, 14, 15, 16]
        
    def test_peek_does_not_remove(self):
        """Test that peek doesn't remove element"""
        heap = Heap(HeapType.MIN, [5, 3, 7])
        
        min_val = heap.heap_peek()
        assert min_val == 3
        assert heap.heap_size() == 3  # Size unchanged
        assert heap.heap_peek() == 3  # Still there
        
    def test_extract_from_empty_heap(self):
        """Test extracting from empty heap raises error"""
        heap = Heap(HeapType.MIN)
        with pytest.raises(IndexError):
            heap.heap_extract()
            
    def test_peek_empty_heap(self):
        """Test peeking into empty heap raises error"""
        heap = Heap(HeapType.MIN)
        with pytest.raises(IndexError):
            heap.heap_peek()
            
    def test_heapify_on_creation(self):
        """Test heap property is maintained when creating with initial data"""
        data = [15, 10, 20, 8, 12, 25, 6]
        heap = Heap(HeapType.MIN, data)
        
        # Root should be minimum
        assert heap.heap_peek() == 6
        
        # Extract all and verify sorted order
        extracted = []
        while not heap.is_heap_empty():
            extracted.append(heap.heap_extract())
        
        assert extracted == sorted(data)
        

class TestMaxHeap:
    """Tests for Max-Heap functionality"""
    
    def test_insert_and_peek_max(self):
        """Test max heap maintains maximum at root"""
        heap = Heap(HeapType.MAX)
        values = [10, 12, 13, 14, 11, 15, 16]
        for val in values:
            heap.heap_insert(val)
        
        assert heap.heap_peek() == 16  # Maximum should be at root
        
    def test_extract_max(self):
        """Test extracting maximum element"""
        heap = Heap(HeapType.MAX, [10, 12, 13, 14, 11, 15, 16])
        
        # Extract in order - should get reverse sorted sequence
        extracted = []
        while not heap.is_heap_empty():
            extracted.append(heap.heap_extract())
        
        assert extracted == [16, 15, 14, 13, 12, 11, 10]
        
    def test_heapify_max_heap(self):
        """Test max heap property is maintained when creating with initial data"""
        data = [15, 10, 20, 8, 12, 25, 6]
        heap = Heap(HeapType.MAX, data)
        
        # Root should be maximum
        assert heap.heap_peek() == 25
        
        # Extract all and verify reverse sorted order
        extracted = []
        while not heap.is_heap_empty():
            extracted.append(heap.heap_extract())
        
        assert extracted == sorted(data, reverse=True)


class TestHeapOperations:
    """Tests for general heap operations"""
    
    def test_clear(self):
        """Test clearing heap"""
        heap = Heap(HeapType.MIN, [1, 2, 3, 4, 5])
        assert heap.heap_size() == 5
        
        heap.heap_clear()
        assert heap.is_heap_empty()
        assert heap.heap_size() == 0
        
    def test_to_list(self):
        """Test getting heap as array"""
        data = [1, 2, 3, 4, 5]
        heap = Heap(HeapType.MIN, data)
        
        heap_array = heap.to_array()
        assert isinstance(heap_array, list)
        assert len(heap_array) == 5
        
        # Modifying returned list should not affect heap
        heap_array.clear()
        assert heap.heap_size() == 5
        
    def test_len_operator(self):
        """Test __len__ operator"""
        heap = Heap(HeapType.MIN, [1, 2, 3])
        assert len(heap) == 3
        
        heap.heap_insert(4)
        assert len(heap) == 4
        
        heap.heap_extract()
        assert len(heap) == 3
        
    def test_bool_operator(self):
        """Test __bool__ operator"""
        heap = Heap(HeapType.MIN)
        assert not bool(heap)  # Empty heap is False
        
        heap.heap_insert(1)
        assert bool(heap)  # Non-empty heap is True
        
    def test_string_representation(self):
        """Test __str__ and __repr__"""
        heap = Heap(HeapType.MIN, [1, 2, 3])
        str_rep = str(heap)
        assert "Min-Heap" in str_rep
        
        heap = Heap(HeapType.MAX, [1, 2, 3])
        str_rep = str(heap)
        assert "Max-Heap" in str_rep


class TestHeapSort:
    """Tests for heap sort algorithm"""
    
    def test_heap_sort_ascending(self):
        """Test heap sort in ascending order"""
        data = [5, 3, 7, 1, 9, 2, 8, 4, 6]
        sorted_data = heap_sort(data)
        assert sorted_data == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
    def test_heap_sort_descending(self):
        """Test heap sort in descending order"""
        data = [5, 3, 7, 1, 9, 2, 8, 4, 6]
        sorted_data = heap_sort(data, reverse=True)
        assert sorted_data == [9, 8, 7, 6, 5, 4, 3, 2, 1]
        
    def test_heap_sort_empty_list(self):
        """Test heap sort with empty list"""
        assert heap_sort([]) == []
        
    def test_heap_sort_single_element(self):
        """Test heap sort with single element"""
        assert heap_sort([5]) == [5]
        
    def test_heap_sort_duplicates(self):
        """Test heap sort with duplicate values"""
        data = [5, 3, 5, 1, 3, 7, 1, 9, 3]
        sorted_data = heap_sort(data)
        assert sorted_data == [1, 1, 3, 3, 3, 5, 5, 7, 9]
        
    def test_heap_sort_already_sorted(self):
        """Test heap sort with already sorted data"""
        data = [1, 2, 3, 4, 5]
        assert heap_sort(data) == [1, 2, 3, 4, 5]
        
    def test_heap_sort_reverse_sorted(self):
        """Test heap sort with reverse sorted data"""
        data = [5, 4, 3, 2, 1]
        assert heap_sort(data) == [1, 2, 3, 4, 5]
        
    def test_heap_sort_strings(self):
        """Test heap sort with strings"""
        data = ["banana", "apple", "cherry", "date"]
        sorted_data = heap_sort(data)
        assert sorted_data == ["apple", "banana", "cherry", "date"]
        
    def test_heap_sort_original_unchanged(self):
        """Test that original list is not modified"""
        data = [5, 3, 7, 1, 9]
        original = data.copy()
        heap_sort(data)
        assert data == original


class TestHeapify:
    """Tests for heapify function"""
    
    def test_heapify_min(self):
        """Test heapify with min heap"""
        data = [15, 10, 20, 8, 12, 25, 6]
        heap = heapify(data, HeapType.MIN)
        
        assert isinstance(heap, Heap)
        assert heap.heap_peek() == 6
        assert heap.heap_size() == 7
        
    def test_heapify_max(self):
        """Test heapify with max heap"""
        data = [15, 10, 20, 8, 12, 25, 6]
        heap = heapify(data, HeapType.MAX)
        
        assert isinstance(heap, Heap)
        assert heap.heap_peek() == 25
        assert heap.heap_size() == 7
        
    def test_heapify_default_is_min(self):
        """Test that heapify defaults to min heap"""
        data = [5, 3, 7, 1]
        heap = heapify(data)
        assert heap.heap_peek() == 1


class TestHeapWithDifferentTypes:
    """Tests for heap with different data types"""
    
    def test_heap_with_floats(self):
        """Test heap with float values"""
        heap = Heap(HeapType.MIN)
        values = [3.14, 2.71, 1.41, 2.23]
        for val in values:
            heap.heap_insert(val)
        
        assert heap.heap_peek() == 1.41
        
    def test_heap_with_strings(self):
        """Test heap with string values"""
        heap = Heap(HeapType.MIN)
        words = ["banana", "apple", "cherry", "date"]
        for word in words:
            heap.heap_insert(word)
        
        assert heap.heap_peek() == "apple"
        
    def test_heap_with_negative_numbers(self):
        """Test heap with negative numbers"""
        heap = Heap(HeapType.MIN, [-5, -1, -10, 0, 3, -7])
        assert heap.heap_peek() == -10
        
        max_heap = Heap(HeapType.MAX, [-5, -1, -10, 0, 3, -7])
        assert max_heap.heap_peek() == 3


class TestHeapProperties:
    """Tests for heap structural properties"""
    
    def test_parent_child_relationships(self):
        """Test that heap maintains proper parent-child relationships"""
        heap = Heap(HeapType.MIN, [10, 12, 13, 14, 11, 15, 16])
        
        # For each non-leaf node, check heap property
        data = heap.to_array()
        for i in range(len(data) // 2):
            left_child_idx = 2 * i + 1
            right_child_idx = 2 * i + 2
            
            if left_child_idx < len(data):
                assert data[i] <= data[left_child_idx]
            
            if right_child_idx < len(data):
                assert data[i] <= data[right_child_idx]
                
    def test_complete_binary_tree_property(self):
        """Test that heap maintains complete binary tree property"""
        heap = Heap(HeapType.MIN)
        
        # Add elements
        for i in range(10):
            heap.heap_insert(i)
        
        # Heap should be stored as complete binary tree
        # All levels except last should be completely filled
        # Last level should be filled from left to right
        assert len(heap) == 10
        
    def test_heap_property_after_operations(self):
        """Test that heap property is maintained after various operations"""
        heap = Heap(HeapType.MIN)
        
        # Insert multiple elements
        for val in [5, 3, 7, 1, 9, 2]:
            heap.heap_insert(val)
            # After each insert, min should be at root
            assert heap.heap_peek() == min(heap.to_array())
        
        # Extract some elements
        heap.heap_extract()
        heap.heap_extract()
        
        # Heap property should still hold
        if not heap.is_heap_empty():
            assert heap.heap_peek() == min(heap.to_array())


class TestEdgeCases:
    """Tests for edge cases"""
    
    def test_single_element_operations(self):
        """Test operations on single-element heap"""
        heap = Heap(HeapType.MIN, [42])
        
        assert heap.heap_peek() == 42
        assert heap.heap_size() == 1
        assert heap.heap_extract() == 42
        assert heap.is_heap_empty()
        
    def test_two_element_heap(self):
        """Test heap with exactly two elements"""
        heap = Heap(HeapType.MIN, [5, 3])
        assert heap.heap_peek() == 3
        
        assert heap.heap_extract() == 3
        assert heap.heap_peek() == 5
        assert heap.heap_extract() == 5
        assert heap.is_heap_empty()
        
    def test_all_equal_elements(self):
        """Test heap with all equal elements"""
        heap = Heap(HeapType.MIN, [5, 5, 5, 5, 5])
        
        while not heap.is_heap_empty():
            assert heap.heap_extract() == 5
            
    def test_large_heap(self):
        """Test heap with many elements"""
        n = 1000
        import random
        data = [random.randint(1, 1000) for _ in range(n)]
        
        heap = Heap(HeapType.MIN, data)
        assert heap.heap_size() == n
        
        # Extract all and verify sorted
        extracted = []
        while not heap.is_heap_empty():
            extracted.append(heap.heap_extract())
        
        assert extracted == sorted(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

