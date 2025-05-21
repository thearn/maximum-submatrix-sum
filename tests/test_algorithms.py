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

def test_max_submatrix_sum_1d():
    matrix = np.array([
        [ 4.6, -4.8,  1.2, -8.4,  9.7, -0.4, -8.3, -5.7, -3.2, -3.7],
        [-4.0,  5.4,  5.6,  5.5,  3.4, -9.5,  5.0, -4.4,  2.9, -2.6],
        [ 4.8,  7.7,  0.6, -6.4, -5.1, -2.4,  5.2,  6.0, -9.8, -5.8],
        [-4.4, -5.2,  8.6, -1.9, -0.4, -7.2,  4.6,  5.5, -5.2,  7.2],
        [ 6.0, -8.9,  2.7,  4.6, -9.8, -9.7, -3.8, -0.6,  3.2,  2.1],
        [-6.5, -5.7,  0.9, -4.5, -5.7, -7.7,  9.6,  4.9, -4.3, -7.4],
        [-4.5,  7.7,  4.5, -4.3,  0.5, -9.9, -0.5, -4.8, -1.8,  8.3],
        [-5.9, -4.4,  8.3,  0.5,  9.4, -2.4, -4.3, -8.8, -0.5, -9.7],
        [ 2.0, -3.9,  0.0, -7.0, -6.9,  3.8, -8.5, -5.0, -5.4, -2.9],
        [ 0.5,  4.6, -7.7,  3.3,  5.3,  0.1,  0.0,  1.5,  8.4,  8.7]
    ])
    ret, max_value, _ = algorithms.brute_submatrix_max(matrix)
    assert abs(max_value - 32.4) < 1e-6
    ret, max_value, _ = algorithms.fft_submatrix_max(matrix)
    assert abs(max_value - 32.4) < 1e-6
    ret, max_value, _ = algorithms.kidane_max_submatrix(matrix)
    assert abs(max_value - 32.4) < 1e-6

def test_max_submatrix_sum_all():
    matrix = np.array([
        [ 4.9, -3.1,  2.1,  9.8, -4.2,  4.2,  7.6, -1.2, -4.7, -9.0],
        [-2.1, -9.1,  9.5,  0.4, -8.4,  0.1, -7.6, -9.2, -5.2,  2.8],
        [ 8.5,  8.5,  0.9,  2.8,  2.8,  6.9,  9.7,  6.5,  1.6,  0.4],
        [-3.7, -4.9, -0.6, -4.3,  7.3, -7.0,  2.0,  9.7, -3.8,  9.5],
        [-1.1,  0.7,  9.4, -0.6, -7.8,  3.7, -6.1, -7.9,  6.6,  1.6],
        [ 3.4,  0.4, -9.2, -9.6, -2.0,  7.0, -3.7,  3.3, -8.0,  5.8],
        [ 6.0, -10.0,  4.4,  6.1,  2.5, -6.4,  1.7, -7.4, -3.4, -0.9],
        [ 6.6,  7.6, -5.1,  6.8,  1.2, -0.5,  3.5,  6.0,  8.7, -3.5],
        [-6.6,  4.1, -9.0,  1.3, -5.6,  6.2,  4.1, -8.0, -0.9, -3.5],
        [ 1.3,  6.1,  2.1, -3.3, -5.1, -0.4, -2.0,  2.0,  5.5, -4.3]
    ])
    ret, max_value, _ = algorithms.brute_submatrix_max(matrix)
    assert abs(max_value - 62.6) < 1e-6
    ret, max_value, _ = algorithms.fft_submatrix_max(matrix)
    assert abs(max_value - 62.6) < 1e-6
    ret, max_value, _ = algorithms.kidane_max_submatrix(matrix)
    assert abs(max_value - 62.6) < 1e-6
