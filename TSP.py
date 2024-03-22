import numpy as np

class TSP:
    def __init__(self, G):
        self.G = G

    def h(self, S):
        return S

    def c(self, H):
        return self.G.walkDistance(self.G.closeWalk(H))

    def mu_initial(self, p):
        return np.zeros((p, self.G.n))

    def relu(self, v):
        return np.vectorize(lambda i: max(0, i))(v)

    def F(self, x, mu, theta_1, theta_2, theta_3, theta_4):
        def f(v):
            r1 = theta_1 * x[v]
            neighbor_v = np.vectorize(lambda i: int(i != v))(self.G.vertices)
            r2 = theta_2 @ mu @ neighbor_v
            r3 = theta_3 @ self.relu(np.outer(theta_4, self.G.dist_matrix[v, :])) @ neighbor_v
            return self.relu(r1 + r2 + r3)
        return f
