#Fold with Integral Objective
import numpy as np
def generate_points_in_simplex(N, c):
    points = np.random.rand(N, c)
    points /= points.sum(axis=1)[:, None]
    return points

def fold(p, L, i, j):
    q = np.insert(np.delete(p, [i, j], axis = 0), min(i,j), p[i] + p[j], axis = 0)

    NewCol = np.concatenate((L[:,i].reshape(-1,1), L[:,j].reshape(-1,1)), axis = 1).max(axis = 1)
    Lt = np.delete(L, [i,j], axis = 1)
    Lt = np.insert(Lt, min(i,j), NewCol, axis = 1)

    return q, Lt

def FoldWithIntegralObjective(L, c, N):    
    # p = [p1p2p3...pN]
    # L = [L]
    # Lp = [Lp1 Lp2 Lp3 ... Lpk]

    M = np.zeros((c,c))
    M += 10**4
    p = generate_points_in_simplex(N, c).T 
    for i in range(c):
        for j in range(i+1, c):
            H_p = np.einsum("ac,cb->ab", L, p).min(axis = 0)
            q, Lt = fold(p, L, i, j)
            H_q = np.einsum("ac,cb->ab", Lt, q).min(axis = 0)
            M[i, j] = (H_q - H_p).mean(axis = 0)

    i_fold, j_fold = np.unravel_index(np.argmin(M), M.shape)
    return (i_fold, j_fold)

L = np.array([[1,0,0],[0,0,1]])
c = 3

i_fold, j_fold = FoldWithIntegralObjective(L, c, 1000)
print(i_fold, j_fold)