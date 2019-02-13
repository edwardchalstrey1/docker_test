# Taken from http://markus-beuckelmann.de/blog/boosting-numpy-blas.html

from __future__ import print_function

import numpy as np
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
size_num = int(size)
rank = comm.Get_rank()

# Let's take the randomness out of random numbers (for reproducibility)
np.random.seed(0)

size = 4096
A, B = np.random.random((size, size)), np.random.random((size, size))
C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
E = np.random.random((int(size / 2), int(size / 4)))
F = np.random.random((int(size / 2), int(size / 2)))
F = np.dot(F, F.T)
G = np.random.random((int(size / 2), int(size / 2)))

if rank == 0:

    # Matrix multiplication
    N = 20
    t = time()
    for i in range(N):
        np.dot(A, B)
    delta = time() - t
    print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
    del A, B

if rank == 1:

    # Vector multiplication
    N = 5000
    t = time()
    for i in range(N):
        np.dot(C, D)
    delta = time() - t
    print('Dotted two vectors of length %d in %0.2f ms.' % (size * 128, 1e3 * delta / N))
    del C, D

if (size_num == 4 and rank == 2) or (size_num == 2 and rank == 0):

    # Singular Value Decomposition (SVD)
    N = 3
    t = time()
    for i in range(N):
        np.linalg.svd(E, full_matrices = False)
    delta = time() - t
    print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
    del E

if (size_num == 4 and rank == 3) or (size_num == 2 and rank == 1):

    # Cholesky Decomposition
    N = 3
    t = time()
    for i in range(N):
        np.linalg.cholesky(F)
    delta = time() - t
    print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))
