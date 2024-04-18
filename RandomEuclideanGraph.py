from EuclideanGraph import EuclideanGraph
import random

class RandomEuclideanGraph(EuclideanGraph):
    def __init__(self, pt_number = (4, 5), xlim = (-3, 3), ylim = (-3, 3)):
        n = random.randint(*pt_number)
        points = [(random.randint(*xlim), random.randint(*ylim)) for i in range(n)]
        super().__init__(points)

