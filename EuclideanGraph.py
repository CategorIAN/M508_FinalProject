from CompleteGraph import CompleteGraph
import numpy as np

class EuclideanGraph(CompleteGraph):
    def __init__(self, points):
        self.points = points
        pt_array = np.array([list(point) for point in points])
        dist = lambda i, j: np.linalg.norm(pt_array[i, :] - pt_array[j, :])
        n = len(pt_array)
        dist_matrix = np.array([[dist(i, j) for j in range(n)] for i in range(n)])
        super().__init__(dist_matrix)

    def __repr__(self):
        return "Graph{}".format(set(self.points))

    def __str__(self):
        return "Graph{}".format(set(self.points))