import numpy as np

class TSP:
    def __init__(self, G, p):
        self.G = G
        self.p = p

    def h(self, S):
        return S

    def c(self, H):
        return self.G.walkDistance(self.G.closeWalk(H))

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
        def go(mu, t):
            if t == T:
                return mu
            else:
                return go(F_func(mu), t + 1)
        return go(np.zeros((self.p, self.G.n)), 0)

