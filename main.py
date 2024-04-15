from RandomEuclideanGraph import RandomEuclideanGraph
from TSP import TSP
import numpy as np
from Theta import RandomTheta

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
        tsp = TSP(G, T, 0.01, 2)
        thetas = RandomTheta(p)
        S = [0]
        print(tsp.policy(thetas, S))
    if i == 4:
        G = RandomEuclideanGraph()
        print(G.points)
        p = 3
        T = 4
        tsp = TSP(G, T)
        thetas = RandomTheta(p)
        S = [0]
        print(tsp.policy(thetas)(S))
if __name__ == '__main__':
    f(2)

