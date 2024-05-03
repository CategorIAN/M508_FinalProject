from EuclideanGraph import EuclideanGraph
import random

class EuclideanGraphDistribution:
    def __init__(self, pt_number_lim = (9, 9), xlim = (-5, 5), ylim = (-5, 5)):
        '''
        :param pt_number: the number of two-dimensional points/vertices to use
        :param xlim: (x_min, x_max) where x_min is the min x coordinate and x_max is the max x coordinate for each point
        :param ylim: (y_min, y_max) where y_min is the min y coordinate and y_max is the max x coordinate for each point
        '''
        self.pt_number_lim, self.xlim, self.ylim = pt_number_lim, xlim, ylim

    def randomGraph(self):
        '''
        :return: a random graph from the distribution
        '''
        n = random.randint(*self.pt_number_lim)
        points = [(random.randint(*self.xlim), random.randint(*self.ylim)) for i in range(n)]
        return EuclideanGraph(points)

