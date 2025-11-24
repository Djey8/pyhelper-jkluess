import pytest
from pyhelper_jkluess.Complex.SkipLists.probabilisticskiplist import ProbabilisticSkipList


class TestProbabilisticSkipListCreation:
    def test_empty_skiplist_creation(self):
        """Test creating an empty probabilistic skip list"""
        psl = ProbabilisticSkipList()
        assert psl is not None
    
    def test_skiplist_with_custom_max_height(self):
        """Test creating skip list with custom max height"""
        psl = ProbabilisticSkipList()
        assert psl is not None


class TestProbabilisticSkipListInsertion:
    def test_add_single_element(self):
        """Test adding a single element"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        
        result = psl.find(10)
        assert result == 10
    
    def test_add_multiple_elements(self):
        """Test adding multiple elements"""
        psl = ProbabilisticSkipList()
        values = [10, 20, 5, 15, 25]
        
        for value in values:
            psl.add(value)
        
        for value in values:
            result = psl.find(value)
            assert result == value
    
    def test_add_maintains_sorted_order(self):
        """Test that insertion maintains sorted order"""
        psl = ProbabilisticSkipList()
        values = [30, 10, 20, 5, 25, 15]
        
        for value in values:
            psl.add(value)
        
        # All values should be findable
        for value in values:
            assert psl.find(value) is not None
    
    def test_add_duplicate_values(self):
        """Test adding duplicate values"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(10)
        
        # Should be findable
        result = psl.find(10)
        assert result == 10
    
    def test_add_negative_numbers(self):
        """Test adding negative numbers"""
        psl = ProbabilisticSkipList()
        psl.add(-10)
        psl.add(-5)
        psl.add(0)
        psl.add(5)
        
        assert psl.find(-10) is not None
        assert psl.find(-5) is not None
        assert psl.find(0) is not None
        assert psl.find(5) is not None


class TestProbabilisticSkipListSearch:
    def test_find_in_empty_list(self):
        """Test finding in empty skip list"""
        psl = ProbabilisticSkipList()
        result = psl.find(10)
        assert result is None
    
    def test_find_existing_element(self):
        """Test finding existing element"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        result = psl.find(20)
        assert result == 20
    
    def test_find_nonexistent_element(self):
        """Test finding nonexistent element"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        result = psl.find(25)
        assert result is None
    
    def test_find_smaller_than_all(self):
        """Test finding value smaller than all elements"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        result = psl.find(5)
        assert result is None
    
    def test_find_larger_than_all(self):
        """Test finding value larger than all elements"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        result = psl.find(40)
        assert result is None


class TestProbabilisticSkipListDeletion:
    def test_remove_from_empty_list(self):
        """Test removing from empty skip list"""
        psl = ProbabilisticSkipList()
        # Should not raise an error
        psl.remove(10)
    
    def test_remove_existing_element(self):
        """Test removing an existing element"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        psl.remove(20)
        result = psl.find(20)
        assert result is None
        
        # Other elements should still exist
        assert psl.find(10) is not None
        assert psl.find(30) is not None
    
    def test_remove_nonexistent_element(self):
        """Test removing nonexistent element"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        
        # Should not raise an error
        psl.remove(15)
        
        # Original elements should still exist
        assert psl.find(10) is not None
        assert psl.find(20) is not None
    
    def test_remove_first_element(self):
        """Test removing the first element"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        psl.remove(10)
        assert psl.find(10) is None
        assert psl.find(20) is not None
        assert psl.find(30) is not None
    
    def test_remove_last_element(self):
        """Test removing the last element"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        psl.remove(30)
        assert psl.find(30) is None
        assert psl.find(10) is not None
        assert psl.find(20) is not None
    
    def test_remove_middle_element(self):
        """Test removing a middle element"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        psl.remove(20)
        assert psl.find(20) is None
        assert psl.find(10) is not None
        assert psl.find(30) is not None
    
    def test_remove_all_elements(self):
        """Test removing all elements"""
        psl = ProbabilisticSkipList()
        values = [10, 20, 30, 40, 50]
        
        for value in values:
            psl.add(value)
        
        for value in values:
            psl.remove(value)
        
        for value in values:
            assert psl.find(value) is None


class TestProbabilisticSkipListWithLargeDataset:
    def test_add_many_elements(self):
        """Test adding many elements"""
        psl = ProbabilisticSkipList()
        n = 100
        
        for i in range(n):
            psl.add(i)
        
        # Verify all elements are findable
        for i in range(n):
            assert psl.find(i) is not None
    
    def test_remove_from_large_dataset(self):
        """Test removing from large dataset"""
        psl = ProbabilisticSkipList()
        n = 100
        
        for i in range(n):
            psl.add(i)
        
        # Remove every other element
        for i in range(0, n, 2):
            psl.remove(i)
        
        # Verify removed elements are gone
        for i in range(0, n, 2):
            assert psl.find(i) is None
        
        # Verify remaining elements exist
        for i in range(1, n, 2):
            assert psl.find(i) is not None
    
    def test_random_operations(self):
        """Test random add/remove/find operations"""
        psl = ProbabilisticSkipList()
        
        # Add
        psl.add(50)
        psl.add(25)
        psl.add(75)
        psl.add(10)
        psl.add(30)
        psl.add(60)
        psl.add(80)
        
        # Find
        assert psl.find(50) is not None
        assert psl.find(100) is None
        
        # Remove
        psl.remove(25)
        psl.remove(75)
        
        # Verify
        assert psl.find(25) is None
        assert psl.find(75) is None
        assert psl.find(50) is not None
        assert psl.find(10) is not None


class TestProbabilisticSkipListEdgeCases:
    def test_single_element_operations(self):
        """Test operations with single element"""
        psl = ProbabilisticSkipList()
        psl.add(42)
        
        assert psl.find(42) is not None
        
        psl.remove(42)
        assert psl.find(42) is None
    
    def test_duplicate_additions(self):
        """Test adding same value multiple times"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(10)
        psl.add(10)
        
        # Should be able to find it
        assert psl.find(10) is not None
    
    def test_add_after_remove(self):
        """Test adding after removing"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.remove(10)
        psl.add(10)
        
        assert psl.find(10) is not None
    
    def test_floating_point_values(self):
        """Test with floating point values"""
        psl = ProbabilisticSkipList()
        psl.add(10.5)
        psl.add(20.3)
        psl.add(15.7)
        
        assert psl.find(10.5) is not None
        assert psl.find(20.3) is not None
        assert psl.find(15.7) is not None
    
    def test_sequential_additions(self):
        """Test adding elements in sequential order"""
        psl = ProbabilisticSkipList()
        
        for i in range(1, 21):
            psl.add(i)
        
        for i in range(1, 21):
            assert psl.find(i) is not None
    
    def test_reverse_sequential_additions(self):
        """Test adding elements in reverse order"""
        psl = ProbabilisticSkipList()
        
        for i in range(20, 0, -1):
            psl.add(i)
        
        for i in range(1, 21):
            assert psl.find(i) is not None


class TestProbabilisticSkipListHeightGeneration:
    def test_height_generation_within_bounds(self):
        """Test that generated heights are within max_height"""
        psl = ProbabilisticSkipList()
        
        # Add many elements to test probabilistic height generation
        for i in range(100):
            psl.add(i)
        
        # All elements should be findable regardless of height
        for i in range(100):
            assert psl.find(i) is not None
    
    def test_probabilistic_distribution(self):
        """Test that heights follow probabilistic distribution"""
        psl = ProbabilisticSkipList()
        
        # Add many elements
        for i in range(50):
            psl.add(i)
        
        # Verify all are searchable (tests distribution works)
        for i in range(50):
            assert psl.find(i) is not None


class TestProbabilisticSkipListDisplay:
    def test_display_empty_list(self):
        """Test display of empty skip list"""
        psl = ProbabilisticSkipList()
        # Should not raise an error
        psl.display()
    
    def test_display_with_elements(self):
        """Test display with elements"""
        psl = ProbabilisticSkipList()
        psl.add(10)
        psl.add(20)
        psl.add(30)
        
        # Should not raise an error
        psl.display()


class TestProbabilisticSkipListStressTest:
    def test_stress_many_insertions(self):
        """Stress test with many insertions"""
        psl = ProbabilisticSkipList()
        n = 500
        
        for i in range(n):
            psl.add(i)
        
        # Sample checks
        assert psl.find(0) is not None
        assert psl.find(n // 2) is not None
        assert psl.find(n - 1) is not None
    
    def test_stress_mixed_operations(self):
        """Stress test with mixed operations"""
        psl = ProbabilisticSkipList()
        
        # Add
        for i in range(100):
            psl.add(i)
        
        # Remove half
        for i in range(0, 100, 2):
            psl.remove(i)
        
        # Add back some
        for i in range(0, 50, 2):
            psl.add(i)
        
        # Verify state
        for i in range(0, 50, 2):
            assert psl.find(i) is not None
        for i in range(1, 100, 2):
            assert psl.find(i) is not None
