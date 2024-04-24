from itertools import product
import numpy as np
from functools import reduce
import random
class CompleteGraph:
    def __init__(self, dist_matrix):
        '''
        :param dist_matrix: n by n matrix w where w[(i, j)] is the weight of the edge (i, j) and where w[(i, i)] = 0
        '''
        m, n = dist_matrix.shape
        if m != n:
            raise AttributeError("Matrix is Not Square")
        self.n = n
        self.vertices = list(range(self.n))
        self.edges = list(product(range(self.n)))
        self.w = (np.ones((n, n)) - np.eye(n)) * dist_matrix
        self.W = self.w.flatten()
        self.neighbors = dict([(i, set(self.filter(self.vertices, lambda j: j != i))) for i in self.vertices])
        self.N = self.neighbor_matrix()
        self.U = self.getU()


    def closeWalk(self, w):
        return w + [w[0]] if len(w) > 0 and w[-1] != w[0] else w

    def randomHamCycle(self):
        return self.closeWalk(random.sample(range(self.n), k = self.n))

    def walkDistance(self, walk):
        def recurse(distance, current_v, toWalk):
            if len(toWalk) == 0:
                return distance
            else:
                next_v, remaining = toWalk[0], toWalk[1:]
                return recurse(distance + self.w[(current_v, next_v)], next_v, remaining)
        return 0 if len(walk) == 0 else recurse(0, walk[0], walk[1:])

    def tourDistance(self, S):
        return self.walkDistance(self.closeWalk(S))

    def filter(self, s, predicate):
        return reduce(lambda v, i: v + [i] if predicate(i) else v, s, [])

    def neighbor_vec(self, v):
        neighbors = self.neighbors[v]
        return np.vectorize(lambda i: int(i in neighbors))(self.vertices).reshape(-1, 1)

    def neighbor_matrix(self):
        return np.concatenate([self.neighbor_vec(v) for v in self.vertices], axis = 1)

    def unit_vec(self, v):
        return np.vectorize(lambda i: int(i == v))(self.vertices).reshape(-1, 1)

    def neighbor_square(self, v):
        return np.outer(self.neighbor_vec(v), self.unit_vec(v).reshape(1, -1))

    def getU(self):
        return np.concatenate([self.neighbor_square(v) for v in self.vertices], axis=0)
