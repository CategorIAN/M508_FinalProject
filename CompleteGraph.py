from itertools import product
import numpy as np
from functools import reduce
import random
class CompleteGraph:
    def __init__(self, dist_matrix):
        m, n = dist_matrix.shape
        if m != n:
            raise AttributeError("Matrix is Not Square")
        self.n = n
        self.vertices = list(range(self.n))
        self.edges = list(product(range(self.n)))
        self.dist_matrix = dist_matrix
        self.neighbors = dict([(i, set(self.filter(self.vertices, lambda j: j != i))) for i in self.vertices])

    def closeWalk(self, w):
        return w + [w[0]]

    def randomHamCycle(self):
        return self.closeWalk(random.sample(range(self.n), k = self.n))

    def walkDistance(self, walk):
        def recurse(distance, current_v, toWalk):
            if len(toWalk) == 0:
                return distance
            else:
                next_v, remaining = toWalk[0], toWalk[1:]
                return recurse(distance + self.dist_matrix[(current_v, next_v)], next_v, remaining)
        return 0 if len(walk) == 0 else recurse(0, walk[0], walk[1:])

    def filter(self, s, predicate):
        return reduce(lambda v, i: v + [i] if predicate(i) else v, s, [])
