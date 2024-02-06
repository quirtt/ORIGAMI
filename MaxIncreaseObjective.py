#Fold with Max-Increase Objective
import numpy as np
from scipy.optimize import minimize

class Max_Increase:
    def __init__(self, c, L, delta, gamma=1):
        self.c = c
        self.L = L
        self.delta = delta #convergence threshold
        self.gamma = gamma #slowdown parameter of objective subgradient

    def X(self,i,j):
        # <- q = Xp ->
        X = np.eye(self.c)
        E_ij,E_jj = (np.zeros((self.c,self.c)), np.zeros((self.c,self.c)))
        E_ij[i, j] = 1
        E_jj[j, j] = 1

        X += E_ij - E_jj
        return X
    
    def Lt(self,i,j):
        L = self.L
        NewCol = np.concatenate((L[:,i].reshape(-1,1), L[:,j].reshape(-1,1)), axis = 1).max(axis = 1)
        Lt = np.delete(L, [i,j], axis = 1)
        Lt = np.insert(Lt, i, NewCol, axis = 1)
        Lt = np.insert(Lt, j, np.zeros(len(L[:, i])), axis = 1)
        return Lt
    
    def objectiveFunction(self, p, pk, i, j):
        i, j = sorted((i,j))
        L = self.L
        gk = (1/self.gamma)*L[np.argmin(L@pk), :]
        return -np.min((self.Lt(i,j)@self.X(i,j))@p) + gk@(p - pk) + np.min(L@pk)

    def probability_constraint(self, p):
        return np.sum(p) - 1
    
    def suboptimalityGap(self, p, i, j):
        return np.min(self.Lt(i,j)@self.X(i,j)@p) - np.min(self.L@p)
    
    def linear_optimizer(self, pk, i, j):
        initial_guess = np.random.rand(self.c)
        initial_guess /= np.sum(initial_guess)
        bounds = [(0,1) for _ in range(self.c)]
        constraints = ({'type':'eq', 'fun': self.probability_constraint})
        result = minimize(self.objectiveFunction, initial_guess, args = (pk, i, j), bounds=bounds, constraints=constraints)

        optimal_p = result.x 
        value = self.suboptimalityGap(optimal_p, i, j)
        Delta = abs(value - self.suboptimalityGap(pk, i, j))
        return (optimal_p, value, Delta)

    def DCOptimizer(self, i, j):
        p = np.random.rand(self.c)
        p /= np.sum(p)
        Delta = np.inf

        while Delta >= self.delta:
            p, value, Delta = self.linear_optimizer(p, i, j)

        return p, value

def FoldWithMaxIncreaseObjective(L, c, delta, gamma):
    #H_l(p) = min_i (L_i*p)
    # subgradient(H_l(p)) = L_i/lambda

    MI = Max_Increase(c,L,delta,gamma)
    M = np.zeros((c,c))
    M += 10**4
    for i in range(c):
        for j in range(i+1, c):
            p, value = MI.DCOptimizer(i,j)
            M[i, j] = value

    i_fold, j_fold = np.unravel_index(np.argmin(M), M.shape)
    return (i_fold, j_fold)

L = np.array([[1,0,0],[0,0,1]])
c = 3
delta = 1/1000
gamma = 1000

i_fold, j_fold = FoldWithMaxIncreaseObjective(L, c, delta, gamma)
print(i_fold, j_fold)