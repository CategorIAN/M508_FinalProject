from RandomEuclideanGraph import RandomEuclideanGraph
from TSP import TSP
import numpy as np
from Theta import RandomThetaObject

def f(i):
    if i == 1:
        G = RandomEuclideanGraph()
        print(G.points)
        Theta = RandomThetaObject(3)
        T, eps, n, alpha = 2, 0.01, 1, 0.01
        tsp = TSP(G, T, eps, n, alpha)
        print(tsp.QLearning(Theta))

if __name__ == '__main__':
    f(1)

