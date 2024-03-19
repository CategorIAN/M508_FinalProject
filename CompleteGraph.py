from itertools import product
import numpy as np
from functools import reduce
class CompleteGraph:
    def __init__(self, dist_matrix):
        m, n = dist_matrix.shape
        if m != n:
            raise AttributeError("Matrix is Not Square")
        self.n = n
        self.vertices = list(range(self.n))
        self.edges = list(product(range(self.n)))
        self.dist_matrix = dist_matrix

    def randomHamCycle(self):
        w = np.random.permutation(self.n)
        return np.concatenate((w, [w[0]]))

    def walkDistance(self, walk):
        def recurse(distance, current_v, toWalk):
            if len(toWalk) == 0:
                return distance
            else:
                next_v, remaining = toWalk[0], toWalk[1:]
                return recurse(distance + self.dist_matrix[(current_v, next_v)], next_v, remaining)
        return 0 if len(walk) == 0 else recurse(0, walk[0], walk[1:])
