import torch
import numpy as np
import scipy
import jax.numpy as jnp
import jax.scipy.linalg as jsp
import jax.lax.linalg as lax
from jax import random
import time

# Generate a random matrix for all frameworks
matrix_size = 80*80  # Change this size for testing scalability
key = random.PRNGKey(0)
A_jax = random.normal(key, shape=(matrix_size, matrix_size))
A_torch = torch.tensor(np.array(A_jax))
A_scipy = np.array(A_jax)  # Reuse JAX-generated data for consistency

# Helper function to time SVD computation
def time_svd(func, name, A, framework="jax"):
    start_time = time.time()
    for i in range(1):
        if framework == "jax":
            func(A)
        elif framework == "torch":
            func(A)
        elif framework == "scipy":
            scipy.sparse.linalg.svds(A, k=80)
    elapsed_time = time.time() - start_time
    print(f"{name}: {elapsed_time:.6f} seconds")
    return elapsed_time

# Timing SVD for each implementation
print("Testing SVD implementations on a random matrix:")

# JAX implementations
time_numpy_jax = time_svd(jnp.linalg.svd, "jax.numpy.linalg.svd", A_jax, framework="jax")
time_scipy_jax = time_svd(jsp.svd, "jax.scipy.linalg.svd", A_jax, framework="jax")
time_lax_jax = time_svd(lax.svd, "jax.lax.linalg.svd", A_jax, framework="jax")

# PyTorch implementation
time_torch = time_svd(torch.linalg.svd, "torch.linalg.svd", A_torch, framework="torch")

# SciPy implementation
time_scipy = time_svd(scipy.sparse.linalg.svds, "scipy.sparse.linalg.svds", A_scipy, framework="scipy")

# Summary
print("\nSummary:")
print(f"jax.numpy.linalg.svd: {time_numpy_jax:.6f} seconds")
print(f"jax.scipy.linalg.svd: {time_scipy_jax:.6f} seconds")
print(f"jax.lax.linalg.svd: {time_lax_jax:.6f} seconds")
print(f"torch.linalg.svd: {time_torch:.6f} seconds")
print(f"scipy.linalg.svd: {time_scipy:.6f} seconds")
