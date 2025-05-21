import sys
import os
import numpy as np
import pytest

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms import brute_submatrix_max, fft_submatrix_max, local_search

# Test cases for local_search
def test_local_search_empty_array():
    A = np.array([[]])
    loc = (slice(0, 0), slice(0, 0))
    # Expecting local_search to handle empty or near-empty slices gracefully
    # Depending on implementation, it might return (0,0) or raise error
    # For now, assume it returns the same slice and sum 0 if slice is empty
    res_loc, res_sum = local_search(A, loc)
    assert res_sum == 0 # Sum of empty slice is 0
    assert res_loc == loc

def test_local_search_single_element_array():
    A = np.array([[5]])
    loc = (slice(0, 1), slice(0, 1))
    res_loc, res_sum = local_search(A, loc)
    assert res_sum == 5
    assert res_loc == loc

def test_local_search_no_change():
    A = np.array([[1, 2], [3, 4]])
    loc = (slice(0, 2), slice(0, 2)) # Whole array
    res_loc, res_sum = local_search(A, loc)
    assert res_sum == 10 # 1+2+3+4
    assert res_loc == loc

def test_local_search_finds_better_sum():
    A = np.array([
        [1, -10, 5],
        [-10, 20, -10],
        [5, -10, 1]
    ])
    # Initial loc is a small part, local search should expand or move to the 20
    initial_loc = (slice(0, 1), slice(0, 1)) # Just the element '1'
    expected_loc = (slice(1, 2), slice(1, 2)) # The element '20'
    res_loc, res_sum = local_search(A, initial_loc)
    assert res_sum == 20
    assert res_loc == expected_loc

# Test cases for brute_submatrix_max
def test_brute_empty_array():
    A = np.array([[]])
    loc, val, t = brute_submatrix_max(A)
    assert val == 0 # Or -np.inf depending on initialization, current code returns 0 for M=0 or N=0
    assert loc == (slice(0,0), slice(0,0))

def test_brute_single_element_positive():
    A = np.array([[5]])
    loc, val, t = brute_submatrix_max(A)
    assert val == 5
    assert loc == (slice(0, 1), slice(0, 1))

def test_brute_single_element_negative():
    A = np.array([[-5]])
    loc, val, t = brute_submatrix_max(A)
    assert val == -5
    assert loc == (slice(0, 1), slice(0, 1))

def test_brute_all_positive():
    A = np.array([[1, 2], [3, 4]])
    loc, val, t = brute_submatrix_max(A)
    assert val == 10 # Sum of all elements
    assert loc == (slice(0, 2), slice(0, 2))

def test_brute_all_negative():
    A = np.array([[-1, -2], [-3, -4]])
    loc, val, t = brute_submatrix_max(A)
    assert val == -1 # Max is the single element -1
    assert loc == (slice(0, 1), slice(0, 1))

def test_brute_mixed_values():
    A = np.array([
        [1, -2, 3],
        [-4, 5, -6],
        [7, -8, 9]
    ])
    # Expected: [[5, -6], [7, -8], [9]] -> sum = 9 (element 9 itself)
    # Or [[5], [7]] -> sum = 12
    # Or [[3], [-6], [9]] -> sum = 9
    # Or [[1, -2, 3], [-4, 5, -6], [7, -8, 9]] -> sum of 7, -8, 9 is 8.
    # Submatrix [5] is 5. Submatrix [9] is 9.
    # Submatrix [[5], [7]] is 12.
    # Submatrix [[3], [9]] is 12.
    # Submatrix [[7, -8, 9]] is 8.
    # Submatrix [[-4, 5], [7, -8]] is 0
    # Submatrix [[5, -6], [-8, 9]] is 0
    # The largest single element is 9.
    # The largest 2x1 is [[5],[7]] sum 12
    # The largest 1x2 is [[7, -8]] sum -1, [[-8, 9]] sum 1
    # The largest 2x2 is [[-4, 5], [7, -8]] sum 0
    # The largest 3x1 is [[1],[-4],[7]] sum 4; [[-2],[5],[-8]] sum -5; [[3],[-6],[9]] sum 6
    # The largest 1x3 is [[7,-8,9]] sum 8
    # The largest submatrix is actually the element 9 itself if we consider single elements.
    # Let's trace:
    # [5] = 5
    # [7] = 7
    # [9] = 9
    # [[5],[-4]] = 1
    # [[5],[7]] = 12. loc = (slice(1,3), slice(1,2))
    # [[-6],[9]] = 3
    # [[3],[-6],[9]] = 6
    # [[1,-2,3],[-4,5,-6],[7,-8,9]]
    # The submatrix [[5], [7]] (column index 1, rows 1 to 2) has sum 5+7=12.
    # The submatrix [[3], [9]] (column index 2, rows 0 and 2) is not contiguous.
    # The submatrix containing just 9 is (slice(2,3), slice(2,3)) sum 9
    # The submatrix [[-4, 5], [7, -8]] sum is 0
    # The submatrix [[1, -2], [-4, 5]] sum is 0
    # The submatrix [[5,-6],[7,-8]] sum is -2
    # The submatrix [[-2,3],[5,-6]] sum is 0
    # The submatrix [[7,-8,9]] sum is 8
    # The submatrix [[-4,5,-6],[7,-8,9]] sum is -3
    # The submatrix [[1,-2,3],[-4,5,-6]] sum is -3
    # The whole matrix sum is -3
    # The submatrix [[5],[7]] is (slice(1,3), slice(1,2)) sum 12
    loc, val, t = brute_submatrix_max(A)
    assert val == 12
    assert loc == (slice(1, 3), slice(1, 2)) # Corresponds to [[-4, 5], [7, -8]] -> [[5],[7]] part

# Test cases for fft_submatrix_max
def test_fft_empty_array():
    A = np.array([[]])
    # fft_submatrix_max calls brute for M < 2 or N < 2.
    # If A is completely empty (0,0) shape, brute_submatrix_max returns (slice(0,0),slice(0,0)), 0
    loc, val, t = fft_submatrix_max(A)
    assert val == 0
    assert loc == (slice(0,0), slice(0,0))

def test_fft_single_element_positive():
    A = np.array([[5]])
    # Falls back to brute
    loc, val, t = fft_submatrix_max(A)
    assert val == 5
    assert loc == (slice(0, 1), slice(0, 1))

def test_fft_single_element_negative():
    A = np.array([[-5]])
    # Falls back to brute
    loc, val, t = fft_submatrix_max(A)
    assert val == -5
    assert loc == (slice(0, 1), slice(0, 1))

def test_fft_all_positive():
    A = np.array([[1, 2], [3, 4]])
    loc, val, t = fft_submatrix_max(A)
    assert val == 10
    assert loc == (slice(0, 2), slice(0, 2))

def test_fft_all_negative():
    A = np.array([[-1, -2], [-3, -4]])
    loc, val, t = fft_submatrix_max(A)
    # The max sum is -1 (the element itself)
    assert val == -1
    assert loc == (slice(0, 1), slice(0, 1))


def test_fft_mixed_values():
    A = np.array([
        [1, -2, 3],
        [-4, 5, -6],
        [7, -8, 9]
    ])
    # Expected from brute: val == 12, loc == (slice(1, 3), slice(1, 2))
    loc, val, t = fft_submatrix_max(A)
    assert val == 12
    assert loc == (slice(1, 3), slice(1, 2))

def test_fft_larger_array():
    A = np.array([
        [10, -5, 0, 15],
        [-20, 25, -10, 5],
        [0, -15, 30, -25],
        [5, 10, -5, 20]
    ])
    # Brute force for this would be:
    # Max sum is likely around the 25 and 30.
    # Submatrix [[25, -10], [-15, 30]] -> 25-10-15+30 = 30
    # Submatrix [[25], [-15]] -> 10
    # Submatrix [[-10], [30]] -> 20
    # Submatrix [[25, -10, 5], [-15, 30, -25]] -> 25-10+5-15+30-25 = 10
    # Submatrix [[10, -5, 0, 15], [-20, 25, -10, 5]] sum = 20
    # Submatrix [[25, -10], [-15, 30], [10, -5]] -> 25-10-15+30+10-5 = 35. loc = (slice(1,4), slice(1,3))
    # This is [[25, -10], [-15, 30], [10, -5]]
    # slice(1,4) means rows 1, 2, 3. slice(1,3) means cols 1, 2.
    # A[1:4, 1:3]
    # [[25, -10],
    #  [-15, 30],
    #  [10, -5]]
    # Sum = 25 - 10 - 15 + 30 + 10 - 5 = 35.

    brute_loc, brute_val, _ = brute_submatrix_max(A)
    fft_loc, fft_val, _ = fft_submatrix_max(A)

    assert fft_val == brute_val
    assert fft_loc == brute_loc
    assert fft_val == 35 # Based on manual calculation above
    assert fft_loc == (slice(1, 4), slice(1, 3))


# Compare brute and fft results on a few more cases
@pytest.mark.parametrize("array_fixture", [
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    np.array([[-1, -2], [-3, 5]]),
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    np.random.randint(-10, 10, size=(5, 5)),
    np.random.randint(-5, 5, size=(10, 8))
])
def test_brute_vs_fft_consistency(array_fixture):
    A = array_fixture
    loc_brute, val_brute, _ = brute_submatrix_max(A)
    loc_fft, val_fft, _ = fft_submatrix_max(A)

    assert val_brute == val_fft, f"Mismatch in max value for array:\n{A}"
    # Slices might be slightly different if multiple submatrices have the same max sum.
    # However, the sum must be identical.
    # For the purpose of this test, if sums are equal, we can assume correctness,
    # as the problem asks for *a* submatrix with max sum.
    # If strict location matching is needed, it's more complex.
    # For now, we check if the sum of the FFT-found location matches the brute max value.
    assert A[loc_fft].sum() == val_brute, f"FFT location sum does not match brute max value for array:\n{A}"

def test_local_search_perturbation_logic():
    # Test case where local_search should shift the window
    A = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 9, 1, 0], # Max sum is centered at 9
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    # Initial location is slightly off-center
    initial_loc = (slice(1, 3), slice(1, 3)) # This is [[1,1],[1,9]] sum = 12
    # Expected location after local search should be centered around 9, e.g. (slice(1,4), slice(1,4)) sum 1+1+1+9+1+1+1 = 15
    # Or just the 3x3 block [[1,1,0],[1,9,1],[1,1,0]] sum 15
    # Or the 3x3 block [[1,1,1],[1,9,1],[1,1,1]] sum 16 (if we consider the full 3x3 around 9)
    # A[1:4,1:4] is [[1,1,0],[1,9,1],[1,1,0]] sum 14
    # A[slice(1,4),slice(1,4)] is [[1,1,0],[1,9,1],[1,1,0]] sum 14
    # The original code's local_search perturbs by 1.
    # If initial_loc is (slice(1,3), slice(1,3)), sum is A[1:3,1:3].sum() = [[1,1],[1,9]].sum() = 12
    # Perturbations:
    # (slice(0,2),slice(0,2)) -> [[0,0],[0,1]] sum 1
    # (slice(1,3),slice(1,3)) -> [[1,1],[1,9]] sum 12 (current)
    # (slice(2,4),slice(2,4)) -> [[1,9],[1,1]] sum 12
    # (slice(1,4),slice(1,4)) -> [[1,1,0],[1,9,1],[1,1,0]] sum 14
    # (slice(0,3),slice(0,3)) -> [[0,0,0],[0,1,1],[0,1,9]] sum 12
    # The largest sum from perturbations of (slice(1,3),slice(1,3)) should be found.
    # The 3x3 matrix centered at '9' is (slice(1,4), slice(1,4))
    # A[1:4, 1:4] = [[1,1,0],[1,9,1],[1,1,0]], sum = 14
    # The problem is that local_search only perturbs the boundaries by +/-1.
    # If loc = (slice(r1,r2),slice(c1,c2)), it tries (slice(r1+i, r2+j), slice(c1+k, c2+l))
    # For initial_loc = (slice(1,3), slice(1,3)), r1=1,r2=3,c1=1,c2=3
    # Trying (slice(1+0, 3+1), slice(1+0, 3+1)) = (slice(1,4), slice(1,4))
    # A[1:4, 1:4] = [[1,1,0],[1,9,1],[1,1,0]], sum = 14. This should be found.

    res_loc, res_sum = local_search(A, initial_loc)
    assert res_sum == 14
    assert res_loc == (slice(1, 4), slice(1, 4))

def test_fft_submatrix_max_small_array_fallback():
    # Test that fft_submatrix_max correctly falls back to brute for small arrays
    A_1x5 = np.array([[1, -2, 3, -4, 5]]) # 1x5
    A_5x1 = np.array([[1], [-2], [3], [-4], [5]]) # 5x1

    loc_brute_1x5, val_brute_1x5, _ = brute_submatrix_max(A_1x5)
    loc_fft_1x5, val_fft_1x5, _ = fft_submatrix_max(A_1x5)
    assert val_fft_1x5 == val_brute_1x5
    assert A_1x5[loc_fft_1x5].sum() == val_brute_1x5

    loc_brute_5x1, val_brute_5x1, _ = brute_submatrix_max(A_5x1)
    loc_fft_5x1, val_fft_5x1, _ = fft_submatrix_max(A_5x1)
    assert val_fft_5x1 == val_brute_5x1
    assert A_5x1[loc_fft_5x1].sum() == val_brute_5x1

def test_fft_submatrix_max_very_small_array_1x1():
    A = np.array([[10]])
    loc_brute, val_brute, _ = brute_submatrix_max(A)
    loc_fft, val_fft, _ = fft_submatrix_max(A)
    assert val_fft == val_brute
    assert val_fft == 10
    assert loc_fft == (slice(0,1), slice(0,1))

def test_fft_submatrix_max_2x1_array():
    A = np.array([[10], [-2]])
    # Brute: max is [10], sum 10
    loc_brute, val_brute, _ = brute_submatrix_max(A)
    # FFT: M=2, N=1. Falls back to brute.
    loc_fft, val_fft, _ = fft_submatrix_max(A)
    assert val_fft == val_brute
    assert val_fft == 10
    assert loc_fft == (slice(0,1), slice(0,1))

def test_fft_submatrix_max_1x2_array():
    A = np.array([[10, -2]])
    # Brute: max is [10], sum 10
    loc_brute, val_brute, _ = brute_submatrix_max(A)
    # FFT: M=1, N=2. Falls back to brute.
    loc_fft, val_fft, _ = fft_submatrix_max(A)
    assert val_fft == val_brute
    assert val_fft == 10
    assert loc_fft == (slice(0,1), slice(0,1))
