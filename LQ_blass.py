import numpy as np
import scipy.linalg as sla
from scipy.linalg import blas

def LQFactorization(A):

    # Check shape of A
    if (A.shape[0] > A.shape[1]):
        print("A must have more cols than rows for LQ factorization.")
        return

    m = A.shape[0]
    n = A.shape[1]
    
    Q = np.zeros((m,n))
    L = np.zeros((n,n))
    
    for i in range(n):
        W = A[i:i+1, :]
        for j in range(i):
                W = W - blas.ddot((A[i:i+1, :]), (Q[j:j+1, :]))*Q[j:j+1, :]
        Q[i:i+1, :] = W/blas.dnrm2(W)
        
    L = blas.dgemm(1.0, A, Q, trans_b=1)
    return (L, Q)

n = 4096
A = np.array(np.random.randn(n,n), order='F')

print(np.round(L))
print(np.round(L@Q))
