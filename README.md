maximum-submatrix-sum
=======================

Python code to find the rectangular submatrix of maximum sum in a given M by N matrix, which is a [common algorithm exercise](http://stackoverflow.com/questions/2643908/getting-the-submatrix-with-maximum-sum).

The solution presented here is unique, though not assymptotically optimal (see below). The heavy-lifting 
is actually performed by the FFT, which can be used to compute all possible sums
of a submatrix of fixed size (thanks for the [Fourier convolution theorem](http://en.wikipedia.org/wiki/Convolution_theorem)). 

By repeating this for all possible submatrix dimensions, this sum is correctly 
maximized.

This solution does not match the efficiency of the best known dynamic programming solution, Kadane’s O(N^3) algorithm. The one shown here is O(N^3 log(N)).
It's more of an academic novelty. I'd be interested to see it benchmarked though.

# Running the code

`algorithms.py` implements the described algorithm, along with a brute force
solution.

`run.py` runs both algorithms on a random 100 by 100 test matrix of integers uniformly sampled from (-100, 100).

The format of the output for each algorithm is:
1. Slice object specifying the maximizing submatrix
2. The resulting sum
3. Running time (seconds)
