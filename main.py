from CompleteGraph import CompleteGraph
from EuclideanGraph import EuclideanGraph
import numpy as np



if __name__ == '__main__':
    #G = CompleteGraph(np.array([[1, 2, 2], [4, 5, 6], [7, 8, 9]]))
    G = EuclideanGraph([(1, 2), (3, 4), (5, 6)])
    walk = G.randomHamCycle()
    print(walk)
    print(G.walkDistance(walk))
