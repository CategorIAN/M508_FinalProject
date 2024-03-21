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
            r2 = theta_2 @ mu @ np.vectorize(lambda i: int(i != v))(self.G.vertices)
            #r3 = theta_3 np.
        return f
