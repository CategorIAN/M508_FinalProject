from EuclideanGraph import EuclideanGraph
import random

class RandomEuclideanGraph(EuclideanGraph):
    def __init__(self, pt_number = (4, 5), xlim = (-3, 3), ylim = (-3, 3)):
        '''
        :param pt_number: the number of two-dimensional points/vertices to use
        :param xlim: (x_min, x_max) where x_min is the min x coordinate and x_max is the max x coordinate for each point
        :param ylim: (y_min, y_max) where y_min is the min y coordinate and y_max is the max x coordinate for each point
        '''
        n = random.randint(*pt_number)
        points = [(random.randint(*xlim), random.randint(*ylim)) for i in range(n)]
        super().__init__(points)

