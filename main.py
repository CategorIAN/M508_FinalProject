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
        tsp = TSP(G)
        x = np.zeros((G.n, 1))
        p = 3
        mu = np.zeros((p, G.n))
        theta_1 = np.random.rand(p, 1)
        #theta_2 = np.random
        print(tsp.relu(np.array([-3, 2, 5, -1])))
if __name__ == '__main__':
    f(2)

