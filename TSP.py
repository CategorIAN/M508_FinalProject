import numpy as np
from functools import reduce
from Theta import Theta

class TSP:
    def __init__(self, G, T):
        self.G = G
        self.T = T

    def c(self, S):
        return - self.G.walkDistance(self.G.closeWalk(S))

    def r(self, S, v):
        return self.c(S+[v]) - self.c(S)

    def x(self, S):
        return np.vectorize(lambda i: int(i in S))(self.G.vertices)

    def relu(self, v):
        return np.vectorize(lambda i: max(0, i))(v)

    def neighbors(self, v):
        return np.vectorize(lambda i: int(i != v))(self.G.vertices).reshape(-1, 1)

    def F(self, Theta, S):
        x = self.x(S)
        def new_mu(mu):
            def g(v):
                neighbors = self.neighbors(v)
                r1 = Theta.theta_1 * x[v]
                r2 = Theta.theta_2 @ mu @ neighbors
                r3 = Theta.theta_3 @ self.relu(np.outer(Theta.theta_4, self.G.dist_matrix[v, :])) @ neighbors
                return self.relu(r1 + r2 + r3)
            return np.concatenate([g(v) for v in self.G.vertices], axis=1)
        return new_mu

    def mu_final(self, Theta, S):
        F_func = self.F(Theta, S)
        mu_recurse = lambda mu, t: mu if t == self.T else mu_recurse(F_func(mu), t + 1)
        return mu_recurse(np.zeros((Theta.p, self.G.n)), 0)

    def Q(self, Theta, S):
        mu = self.mu_final(Theta, S)
        r1 = Theta.theta_6 @ mu @ np.ones((self.G.n, 1))
        def f(v):
            r2 = Theta.theta_7 @ mu[:, [v]]
            return (Theta.theta_5.reshape(1, -1) @ self.relu(np.concatenate([r1, r2], axis=0)))[0,0]
        return f

    def policy(self, Theta):
        def f(S):
            Q = self.Q(Theta, S)
            vQs = [(v, Q(v)) for v in set(self.G.vertices).difference(S)]
            return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, vQs, (None, None))[0]
        return f

