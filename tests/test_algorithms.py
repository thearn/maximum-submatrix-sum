import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import algorithms

def test_brute_force():
    # Test a simple 2x2 matrix.
    matrix = np.array([
        [1, -2],
        [-3, 4]
    ])
    # The best submatrix is expected to have a sum of 4.
    _, max_value, _ = algorithms.brute_submatrix_max(matrix)
    assert max_value == 4

def test_fft():
    # Using the same matrix as brute_force.
    matrix = np.array([
        [1, -2],
        [-3, 4]
    ])
    _, max_value, _ = algorithms.fft_submatrix_max(matrix)
    assert max_value == 4

def test_kidane():
    # Test a 3x3 matrix.
    matrix = np.array([
        [1, -2, 3],
        [-4, 5, -6],
        [7, -8, 9]
    ])
    # The best submatrix is expected to have a sum of 9.
    _, max_value, _ = algorithms.kidane_max_submatrix(matrix)
    assert max_value == 9
