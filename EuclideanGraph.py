from CompleteGraph import CompleteGraph
import numpy as np
from TSP_HK import TSP_HK

class EuclideanGraph(CompleteGraph):
    def __init__(self, points, shortest_cycle = None, distance = None):
        '''
        :param points: the two-dimensional points to use for the graph
        '''
        self.points = points
        self.shortest_cycle = shortest_cycle
        self.distance = distance
        pt_array = np.array([list(point) for point in points])
        dist = lambda i, j: np.linalg.norm(pt_array[i, :] - pt_array[j, :])
        n = len(pt_array)
        dist_matrix = np.array([[dist(i, j) for j in range(n)] for i in range(n)])
        super().__init__(dist_matrix)

    def __repr__(self):
        return "Graph{}".format(set(self.points))

    def __str__(self):
        return "Graph{}".format(set(self.points))

    def thisWithShortestCycle(self):
        S = TSP_HK().calculateWalk(self)
        d = self.walkDistance(self.closeWalk(S))
        return EuclideanGraph(self.points, shortest_cycle = S, distance = d)

