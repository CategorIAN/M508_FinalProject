import numpy as np
from functools import reduce
from Theta import Theta
import random

class TSP:
    def __init__(self, G, T, eps, n):
        self.G = G
        self.T = T
        self.eps = eps
        self.n = n
        self.N = self.neighbor_matrix()
        self.W = G.dist_matrix.flatten()
        self.P = self.getP()

    def c(self, S):
        return - self.G.walkDistance(self.G.closeWalk(S))

    def r(self, S, v):
        return self.c(S+[v]) - self.c(S)

    def x(self, S):
        return np.vectorize(lambda i: int(i in S))(self.G.vertices)

    def relu(self, v):
        return np.vectorize(lambda i: max(0, i))(v)

    def neighbor_vec(self, v):
        return np.vectorize(lambda i: int(i != v))(self.G.vertices).reshape(-1, 1)

    def neighbor_matrix(self):
        return np.concatenate([self.neighbor_vec(v) for v in self.G.vertices], axis = 1)

    def unit_vec(self, v):
        return np.vectorize(lambda i: int(i == v))(self.G.vertices)

    def neighbor_square(self, v):
        return np.outer(self.neighbor_vec(v), self.unit_vec(v))

    def getP(self):
        return np.concatenate([self.neighbor_square(v) for v in self.G.vertices], axis=0)

    def F(self, Theta, S):
        x = self.x(S)
        def new_mu(mu):
            r1 = np.outer(Theta.theta_1, x)
            r2 = Theta.theta_2 @ mu @ self.N
            r3 = Theta.theta_3 @ self.relu(np.outer(Theta.theta_4, self.W)) @ self.P
            return self.relu(r1 + r2 + r3)
        return new_mu

    def mu_final(self, Theta, S):
        F_func = self.F(Theta, S)
        mu_recurse = lambda mu, t: mu if t == self.T else mu_recurse(F_func(mu), t + 1)
        return mu_recurse(np.zeros((Theta.p, self.G.n)), 0)

    def Q_vec(self, Theta, S):
        mu = self.mu_final(Theta, S)
        r1 = Theta.theta_5a.reshape(1, -1) @ self.relu(Theta.theta_6 @ mu @ np.ones((self.G.n, self.G.n)))
        r2 = Theta.theta_5b.reshape(1, -1) @ self.relu(Theta.theta_7 @ mu)
        return (r1 + r2)[0]

    def S_not(self, S):
        return set(self.G.vertices).difference(S)

    def policy(self, Theta, S, S_not = None):
        S_not = self.S_not(S) if S_not is None else S_not
        Qvec = self.Q_vec(Theta, S)
        print(Qvec)
        vQs = [(v, Qvec[v]) for v in S_not]
        return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, vQs, (None, None))

    def vBest(self, Theta, S, S_not = None):
        return self.policy(Theta, S, S_not)[0]

    def QBest(self, Theta, S, S_not = None):
        return self.policy(Theta, S, S_not)[1]

    def updated_S(self, Theta):
        def f(S):
            S_not = self.S_not(S)
            v = np.random.choice(S_not) if np.random.rand() < self.eps else self.policy(Theta, S, S_not)
            return S + [v]
        return f

    def updated_C(self, S):
        w = self.G.dist_matrix
        def f(C):
            t = len(C)
            c = C[-1] + w[(S[t-1], S[0])] - w[(S[t-1], S[t])] - w[(S[t], S[0])]
            return C + [c]
        return f

    def R(self, C, t, n):
        return C[t + n - 1] - C[t - 1]

    def updated_M(self, S, C):
        t = len(S)
        def f(M):
            return M + [(S[:(t-self.n)], S[t-self.n], self.R(C, t-self.n, self.n), S)] if t >= self.n else M
        return f

    def B(self, m, M):
        return random.sample(M, m)

    def updated_Theta(self, predicate, M):
        def f(Theta):
            if predicate:
                B = self.B(1, M)[0]

    def QLearning(self, M, Theta, S, C):
        S_new = self.updated_S(Theta)(S)
        C_new = self.updated_C(S_new)(C)
        M_new = self.updated_M(S, C)(M)





