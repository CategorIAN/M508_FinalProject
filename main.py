from TSP_Q import TSP_Q
from RegularPolygon import RandomRegularPolygon, RegularPolygon
from Analysis import Analysis
from TSP_HK import TSP_HK
from RandomEuclideanGraph import RandomEuclideanGraph

def f(i):
    if i == 1:
        p, T, eps, n, alpha, beta, gamma = 3, 2, 0.01, 1, 0.01, 5, 0.9
        tsp = TSP(p, T, eps, n, alpha, beta, gamma)
        Gs = [RandomRegularPolygon() for i in range(9)]
        G_S_list, Theta_final = tsp.QLearning(Gs)
        print(G_S_list)
    if i == 2:
        hyp_range_dict = {"p": [3, 4], "T": [2, 3], "eps": [0.01, 0.1], "n": [2, 3],
                    "alpha": [0.01, 0.1], "beta": [5, 10], "gamma": [0.9, 1]}
        A = Analysis(10, hyp_range_dict)
        A.getErrorDf(append = True)
    if i == 3:
        G = RandomEuclideanGraph()
        tsp = TSP_HK()
        print(tsp.calculateWalk(G))


if __name__ == '__main__':
    f(3)

