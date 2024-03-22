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
        tsp = TSP(G, p)
        x = np.zeros((G.n, 1))
        theta_1 = np.random.rand(p, 1)
        theta_2 = np.random.rand(p, p)
        theta_3 = np.random.rand(p, p)
        theta_4 = np.random.rand(p, 1)
        print(tsp.struc2vec(x, theta_1, theta_2, theta_3, theta_4, 4))
if __name__ == '__main__':
    f(2)

