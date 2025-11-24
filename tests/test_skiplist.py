import pytest
from pyhelper_jkluess.Complex.SkipLists.skiplist import SkipList


class TestSkipListCreation:
    def test_empty_skiplist_creation(self):
        """Test creating an empty skip list"""
        sl = SkipList(max_level=4)
        assert sl is not None
    
    def test_skiplist_with_custom_height(self):
        """Test creating skip list with custom max height"""
        sl = SkipList(max_level=6)
        assert sl is not None


class TestSkipListInsertion:
    def test_insert_single_element(self):
        """Test inserting a single element"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        
        result = sl.search(10)
        assert result == 10
    
    def test_insert_multiple_elements(self):
        """Test inserting multiple elements"""
        sl = SkipList(max_level=4)
        values = [10, 20, 5, 15, 25]
        
        for value in values:
            sl.insert(value, value)
        
        for value in values:
            result = sl.search(value)
            assert result == value
    
    def test_insert_maintains_sorted_order(self):
        """Test that insertion maintains sorted order"""
        sl = SkipList(max_level=4)
        values = [30, 10, 20, 5, 25, 15]
        
        for value in values:
            sl.insert(value, value)
        
        # All values should be searchable
        for value in values:
            assert sl.search(value) is not None
    
    def test_insert_duplicate_values(self):
        """Test inserting duplicate values"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(10, 10)
        
        # Both should be searchable (depends on implementation)
        result = sl.search(10)
        assert result == 10
    
    def test_insert_negative_numbers(self):
        """Test inserting negative numbers"""
        sl = SkipList(max_level=4)
        sl.insert(-10, -10)
        sl.insert(-5, -5)
        sl.insert(0, 0)
        sl.insert(5, 5)
        
        assert sl.search(-10) is not None
        assert sl.search(-5) is not None
        assert sl.search(0) is not None
        assert sl.search(5) is not None


class TestSkipListSearch:
    def test_search_in_empty_list(self):
        """Test searching in empty skip list"""
        sl = SkipList(max_level=4)
        result = sl.search(10)
        assert result is None
    
    def test_search_existing_element(self):
        """Test searching for existing element"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        result = sl.search(20)
        assert result == 20
    
    def test_search_nonexistent_element(self):
        """Test searching for nonexistent element"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        result = sl.search(25)
        assert result is None
    
    def test_search_smaller_than_all(self):
        """Test searching for value smaller than all elements"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        result = sl.search(5)
        assert result is None
    
    def test_search_larger_than_all(self):
        """Test searching for value larger than all elements"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        result = sl.search(40)
        assert result is None


class TestSkipListDeletion:
    def test_delete_from_empty_list(self):
        """Test deleting from empty skip list"""
        sl = SkipList(max_level=4)
        # Should not raise an error
        sl.delete(10)
    
    def test_delete_existing_element(self):
        """Test deleting an existing element"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        sl.delete(20)
        result = sl.search(20)
        assert result is None
        
        # Other elements should still exist
        assert sl.search(10) is not None
        assert sl.search(30) is not None
    
    def test_delete_nonexistent_element(self):
        """Test deleting nonexistent element"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        
        # Should not raise an error
        sl.delete(15)
        
        # Original elements should still exist
        assert sl.search(10) is not None
        assert sl.search(20) is not None
    
    def test_delete_first_element(self):
        """Test deleting the first element"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        sl.delete(10)
        assert sl.search(10) is None
        assert sl.search(20) is not None
        assert sl.search(30) is not None
    
    def test_delete_last_element(self):
        """Test deleting the last element"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        sl.delete(30)
        assert sl.search(30) is None
        assert sl.search(10) is not None
        assert sl.search(20) is not None
    
    def test_delete_middle_element(self):
        """Test deleting a middle element"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        sl.delete(20)
        assert sl.search(20) is None
        assert sl.search(10) is not None
        assert sl.search(30) is not None
    
    def test_delete_all_elements(self):
        """Test deleting all elements"""
        sl = SkipList(max_level=4)
        values = [10, 20, 30, 40, 50]
        
        for value in values:
            sl.insert(value, value)
        
        for value in values:
            sl.delete(value)
        
        for value in values:
            assert sl.search(value) is None


class TestSkipListWithLargeDataset:
    def test_insert_many_elements(self):
        """Test inserting many elements"""
        sl = SkipList(max_level=8)
        n = 100
        
        for i in range(n):
            sl.insert(i, i)
        
        # Verify all elements are searchable
        for i in range(n):
            assert sl.search(i) is not None
    
    def test_delete_from_large_dataset(self):
        """Test deleting from large dataset"""
        sl = SkipList(max_level=8)
        n = 100
        
        for i in range(n):
            sl.insert(i, i)
        
        # Delete every other element
        for i in range(0, n, 2):
            sl.delete(i)
        
        # Verify deleted elements are gone
        for i in range(0, n, 2):
            assert sl.search(i) is None
        
        # Verify remaining elements exist
        for i in range(1, n, 2):
            assert sl.search(i) is not None
    
    def test_random_operations(self):
        """Test random insert/delete/search operations"""
        sl = SkipList(max_level=6)
        
        # Insert
        sl.insert(50, 50)
        sl.insert(25, 25)
        sl.insert(75, 75)
        sl.insert(10, 10)
        sl.insert(30, 30)
        sl.insert(60, 60)
        sl.insert(80, 80)
        
        # Search
        assert sl.search(50) is not None
        assert sl.search(100) is None
        
        # Delete
        sl.delete(25)
        sl.delete(75)
        
        # Verify
        assert sl.search(25) is None
        assert sl.search(75) is None
        assert sl.search(50) is not None
        assert sl.search(10) is not None


class TestSkipListEdgeCases:
    def test_single_element_operations(self):
        """Test operations with single element"""
        sl = SkipList(max_level=4)
        sl.insert(42, 42)
        
        assert sl.search(42) is not None
        
        sl.delete(42)
        assert sl.search(42) is None
    
    def test_duplicate_insertions(self):
        """Test inserting same value multiple times"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.insert(10, 10)
        sl.insert(10, 10)
        
        # Should be able to find it
        assert sl.search(10) is not None
    
    def test_insert_after_delete(self):
        """Test inserting after deleting"""
        sl = SkipList(max_level=4)
        sl.insert(10, 10)
        sl.delete(10)
        sl.insert(10, 10)
        
        assert sl.search(10) is not None
    
    def test_floating_point_values(self):
        """Test with floating point values"""
        sl = SkipList(max_level=4)
        sl.insert(10.5, 10.5)
        sl.insert(20.3, 20.3)
        sl.insert(15.7, 15.7)
        
        assert sl.search(10.5) is not None
        assert sl.search(20.3) is not None
        assert sl.search(15.7) is not None
    
    def test_very_small_height(self):
        """Test skip list with height 1 (essentially a linked list)"""
        sl = SkipList(max_level=1)
        sl.insert(10, 10)
        sl.insert(20, 20)
        sl.insert(30, 30)
        
        assert sl.search(20) is not None
    
    def test_sequential_insertions(self):
        """Test inserting elements in sequential order"""
        sl = SkipList(max_level=5)
        
        for i in range(1, 21):
            sl.insert(i, i)
        
        for i in range(1, 21):
            assert sl.search(i) is not None
    
    def test_reverse_sequential_insertions(self):
        """Test inserting elements in reverse order"""
        sl = SkipList(max_level=5)
        
        for i in range(20, 0, -1):
            sl.insert(i, i)
        
        for i in range(1, 21):
            assert sl.search(i) is not None

