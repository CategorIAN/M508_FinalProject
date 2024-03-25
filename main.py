from RandomEuclideanGraph import RandomEuclideanGraph
from TSP import TSP
import numpy as np

def f(i):
    if i == 1:
        G = RandomEuclideanGraph()
        print(G.points)
        walk = G.randomHamCycle()
        print(walk)
        print(G.walkDistance(walk))
        print(G.neighbors)
    if i == 2:
        G = RandomEuclideanGraph()
        print(G.points)
        p = 3
        T = 4
        tsp = TSP(G, p, T)
        x = np.zeros((G.n, 1))
        theta_1 = np.random.rand(p, 1)
        theta_2 = np.random.rand(p, p)
        theta_3 = np.random.rand(p, p)
        theta_4 = np.random.rand(p, 1)
        theta_5 = np.random.rand(2 * p, 1)
        theta_6 = np.random.rand(p, p)
        theta_7 = np.random.rand(p, p)
        mu = tsp.struc2vec(x, theta_1, theta_2, theta_3, theta_4, 4)
        print(tsp.Q(mu, theta_5, theta_6, theta_7)(0))
    if i == 3:
        G = RandomEuclideanGraph()
        print(G.points)
        p = 3
        tsp = TSP(G, p)
        S = [0, 1]
        print(tsp.r(S, 2))
    if i == 4:
        G = RandomEuclideanGraph()
        print(G.points)
        p = 3
        T = 4
        tsp = TSP(G, p, T)
        theta_1 = np.random.rand(p, 1)
        theta_2 = np.random.rand(p, p)
        theta_3 = np.random.rand(p, p)
        theta_4 = np.random.rand(p, 1)
        theta_5 = np.random.rand(2 * p, 1)
        theta_6 = np.random.rand(p, p)
        theta_7 = np.random.rand(p, p)
        S = [0]
        print(tsp.policy(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7)(S))
if __name__ == '__main__':
    f(4)

