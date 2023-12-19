import numpy as np
def FoldWithVertexObjective(L, c):
    M = np.zeros((c,c))
    M += 10**4
    for i in range(c):
        for j in range(i+1, c):
            zeroC = np.zeros(c, dtype=int)
            p, q = (zeroC, zeroC)
            p[i] = 1
            q[j] = 1
            H_p = np.min(L@p)
            H_q = np.min(L@q)
            M[i,j] = H_q - H_p

    i_fold, j_fold = np.unravel_index(np.argmin(M), M.shape)
    return (i_fold, j_fold)

L = np.array([[1,0,0],[0,0,1]])
c = 3

i_fold, j_fold = FoldWithVertexObjective(L, c)
print(i_fold, j_fold)