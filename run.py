from algorithms import brute_submatrix_max, fft_submatrix_max
import numpy as np

# Set matrix dimensions (rows, columns)
M, N = 64, 64

# Generate MxN matrix of random integers
A = np.random.randint(-100, 100, size=(M, N))

# Test each algorithm
# output format: maximizing subarray slice specification, maximum sum
# value, running time
print()
print("Running FFT algorithm:")
print(fft_submatrix_max(A))
print("Running brute force algorithm:")
print(brute_submatrix_max(A))
