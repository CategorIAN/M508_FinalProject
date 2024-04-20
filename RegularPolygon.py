from EuclideanGraph import EuclideanGraph
import numpy as np
import random

class RegularPolygon (EuclideanGraph):
    def __init__(self, n, radius):
        theta = 2 * np.pi / n
        points = [(radius * np.cos(j * theta), radius * np.sin(j * theta)) for j in range(n)]
        super().__init__(points)
        self.perimeter = n * radius * np.sqrt(2 * (1 - np.cos(theta)))

class RandomRegularPolygon (RegularPolygon):
    def __init__(self, n_lim = (4, 10), radius_lim = (10, 30)):
        n, radius = random.randint(*n_lim), random.randint(*radius_lim)
        super().__init__(n, radius)