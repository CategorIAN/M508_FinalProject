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
        G = RandomEuclideanGraph(xlim=(0,0))
        print(G.points)
        tsp = TSP(G)
        print(tsp.relu(np.array([-3, 2, 5, -1])))

if __name__ == '__main__':
    f(2)

