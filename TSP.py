import numpy as np
from functools import reduce

class TSP:
    def __init__(self, G, p, T):
        self.G = G
        self.p = p
        self.T = T

    def h(self, S):
        return S

    def c(self, H):
        return - self.G.walkDistance(self.G.closeWalk(H))

    def x(self, S):
        return np.vectorize(lambda i: int(i in S))(self.G.vertices)

    def mu_initial(self, p):
        return np.zeros((p, self.G.n))

    def relu(self, v):
        return np.vectorize(lambda i: max(0, i))(v)

    def F(self, x, theta_1, theta_2, theta_3, theta_4):
        def f(mu):
            def g(v):
                r1 = theta_1 * x[v]
                neighbor_v = np.vectorize(lambda i: int(i != v))(self.G.vertices).reshape(-1, 1)
                r2 = theta_2 @ mu @ neighbor_v
                r3 = theta_3 @ self.relu(np.outer(theta_4, self.G.dist_matrix[v, :])) @ neighbor_v
                return self.relu(r1 + r2 + r3)
            return np.concatenate([g(v) for v in self.G.vertices], axis=1)
        return f

    def struc2vec(self, x, theta_1, theta_2, theta_3, theta_4, T):
        F_func = self.F(x, theta_1, theta_2, theta_3, theta_4)
        go = lambda mu, t: mu if t == T else go(F_func(mu), t + 1)
        return go(np.zeros((self.p, self.G.n)), 0)

    def Q(self, mu, theta_5, theta_6, theta_7):
        def f(v):
            r1 = theta_6 @ mu @ np.ones((self.G.n, 1))
            r2 = theta_7 @ mu[:, [v]]
            return (theta_5.reshape(1, -1) @ self.relu(np.concatenate([r1, r2], axis=0)))[0,0]
        return f

    def r(self, S, v):
        return self.c(S+[v]) - self.c(S)

    def policy(self, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7):
        def f(S):
            x = self.x(S)
            mu = self.struc2vec(x, theta_1, theta_2, theta_3, theta_4, self.T)
            Q = self.Q(mu, theta_5, theta_6, theta_7)
            vQs = [(v, Q(v)) for v in set(self.G.vertices).difference(S)]
            return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, vQs, (None, None))[0]
        return f

