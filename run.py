from algorithms import brute_submatrix_max, fft_submatrix_max, kidane_max_submatrix
import numpy as np
from typing import Tuple, Any

# Define type aliases for clarity
SlicePair = Tuple[slice, slice]
Numeric = Any # Matches algorithms.py

# Set matrix dimensions (rows, columns)
M: int = 64
N: int = 64

# Generate MxN matrix of random integers
A: np.ndarray = np.random.randint(-100, 100, size=(M, N))

# Test each algorithm
# output format: maximizing subarray slice specification, maximum sum
# value, running time
print()
print("Running brute force algorithm:")
print(brute_submatrix_max(A))
print("Running FFT algorithm:")
print(fft_submatrix_max(A))
print("Running Kidane algorithm:")
print(kidane_max_submatrix(A))
