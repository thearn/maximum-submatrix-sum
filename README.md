This is a python code which implements a new algorithm to find the rectangular submatrix of maximum sum in a given M by N matrix, which is a 
[common algorithm exercise](http://stackoverflow.com/questions/2643908/getting-the-submatrix-with-maximum-sum).

The solution presented here is unique, though not asymptotically optimal (see below). The heavy-lifting 
is actually performed by the FFT, which can be used to compute all possible sums
of a submatrix of fixed size (thanks for the [Fourier convolution theorem](http://en.wikipedia.org/wiki/Convolution_theorem)). 
This makes it a divide-and-conquer algorithm.

By computing this convolution for all possible submatrix dimensions, the maximum sum can be determined.

This solution does not match the efficiency of the best known dynamic programming solution, Kadane’s O(N^3) algorithm 
(here we let M = N). The one shown here is O(N^3 log(N)) (again, for M = N).
It's more of a toy exercise / academic novelty. 

# Derivation

For simplicity, let A be a real-valued N by N matrix.

The submatrix maximization problem is to find the four integers 

<img src="https://raw.github.com/thearn/maximum-submatrix-sum/master/images/7.png" height="20px" />&nbsp; and &nbsp;<img src="https://raw.github.com/thearn/maximum-submatrix-sum/master/images/8.png" height="20px" />

that maximize:

<img src="https://raw.github.com/thearn/maximum-submatrix-sum/master/images/1.png" height="75px" />

Define m and n as

<img src="https://raw.github.com/thearn/maximum-submatrix-sum/master/images/2.png" height="20px" />

and K to be the matrix of ones

<img src="https://raw.github.com/thearn/maximum-submatrix-sum/master/images/3.png" height="100px" />

Now, consider the [discrete convolution](http://en.wikipedia.org/wiki/Convolution) of the matrices A and K

<img src="https://raw.github.com/thearn/maximum-submatrix-sum/master/images/4.png" height="75px" />

That is, the elements of the convolution of A and K are the sums of all possible m by n contiguous submatrices of A. 
Finally, let K_0 be a zero-padded representation of K, so that the dimensions of A and K are matched. The convolution
operation will still provide the required sums, and can be efficienctly computed by the 
[Fourier convolution theorem](http://en.wikipedia.org/wiki/Convolution_theorem)):

<img src="https://raw.github.com/thearn/maximum-submatrix-sum/master/images/5.png" height="35px" />

where ^ denotes application of the 2D FFT and the dot denotes component-wise multiplication.

So for a candidate submatrix dimension m-by-n, multiply the elements of an FFT of an m-by-n matrix of ones with the FFT 
of A, then compute the inverse 2D FFT of the result to obtain the m-by-n submatrix 
sums of A, in all possible locations. By recording the maximum value & corresponding location, and repeating this for
all possible m and n, we can solve the problem.

This requires taking the FFT of A at the beginning, then for each m and n, taking the FFT of K, element-wise 
multiplying two matrices, taking the inverse FFT of the result, and finding the maximum value in the convolution.
Overall, the complexity is

<img src="https://raw.github.com/thearn/maximum-submatrix-sum/master/images/6.png" height="30px" />

Note that the convolution theorem assumes 
[periodic boundary conditions](http://en.wikipedia.org/wiki/Periodic_boundary_conditions) for the convolution operation. This means that
the simplest implementation of this algorithm technically allows for a submatrix that is wrapped around A. In python
syntax, this would correspond to allowing negative array indices. This can easily be remedied while traversing the
convolution matrix for the maximum value - a mask can be applied to elements of the convolution corresponding to 
wrapped submatrices.

# Running the code

Scipy is required (for the FFT module).

[algorithms.py](https://github.com/thearn/maximum-submatrix-sum/blob/master/algorithms.py) implements the described algorithm, along with a brute force
solution.

[run.py](https://github.com/thearn/maximum-submatrix-sum/blob/master/run.py) runs both algorithms on a random 100 by 100 test matrix of integers uniformly sampled from (-100, 100).

The format of the output for each algorithm is:

  1.  Slice object specifying the maximizing submatrix
  2.  The resulting sum
  3.  Running time (seconds)

The output on my machine gives:

```bash
> python run.py

Running FFT algorithm:
((slice(33, 60, None), slice(12, 76, None)), 5415, 4.183000087738037)
Running brute force algorithm:
((slice(33, 60, None), slice(12, 76, None)), 5415, 29.853000164031982)
```

The FFT algorithm here took 4.18 seconds, while the brute force algorithm took almost 30. 
