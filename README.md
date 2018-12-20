Sparse Locally Linear Embedding

This repository contains MATLAB code related to the following article:

L. Ziegelmeier, M. Kirby, C. Peterson, "Sparse Locally Linear Embedding,"
Procedia Computer Science, 108C (2017), pages 635-644.

The file SparseLLE.m is the main piece of code which computes a lower-dimensional embedding by finding nearest neighbors of each point, determining a sparse weight vector of each of these nearest neighbors, and then uses this weight vector to find a lower-dimensional embedding by solving an eigenvector problem. The sparsity induced in the weights automatically selects an appropriate nearest neighbors for each point.

The file PDIPAQuad.m is called in the SparseLLE.m code. This solves the formulated quadratic program via the primal dual inter point algorithm. Note that this solves the full system of equations, however, reduced systems can speed up computations (although might induce instability).

