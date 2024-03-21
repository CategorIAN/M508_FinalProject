from RandomEuclideanGraph import RandomEuclideanGraph
from TSP import TSP

def f(i):
    if i == 1:
        G = RandomEuclideanGraph()
        print(G.points)
        walk = G.randomHamCycle()
        print(walk)
        print(G.walkDistance(walk))
    if i == 2:
        G = RandomEuclideanGraph(xlim=(0,0))
        print(G.points)
        tsp = TSP(G)
        print(tsp.c([0, 1]))

if __name__ == '__main__':
    f(2)

