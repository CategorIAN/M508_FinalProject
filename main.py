from RandomEuclideanGraph import RandomEuclideanGraph
from TSP import TSP
from Theta import RandomThetaObject

def f(i):
    if i == 1:
        G = RandomEuclideanGraph()
        print(G.points)
        Theta = RandomThetaObject(3)
        T, eps, n, alpha = 2, 0.01, 1, 0.01
        tsp = TSP(T, eps, n, alpha)
        G_S_list, (Theta_final, M_final) = tsp.QLearning(Theta, 5)
        print(G_S_list)

if __name__ == '__main__':
    f(1)

