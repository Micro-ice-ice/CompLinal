import numpy as np
import scipy.linalg as sla

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
                W = W - np.dot((A[i:i+1, :]),np.transpose(Q[j:j+1, :]))*Q[j:j+1, :]
        Q[i:i+1, :] = W/sla.norm(W)
        
    L = A@Q.transpose()
    return (L, Q)

n = 1000
A = np.array(np.random.randn(n,n), order='F')
[L, Q] = LQFactorization(A)

print(np.round(L))
print(np.round(L@Q))