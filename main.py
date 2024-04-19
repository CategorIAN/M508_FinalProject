from TSP import TSP

def f(i):
    if i == 1:
        T, eps, n, alpha, p = 2, 0.01, 1, 0.01, 3
        tsp = TSP(T, eps, n, alpha, p)
        G_S_list, (Theta_final, M_final) = tsp.QLearning(5)
        print(G_S_list)

if __name__ == '__main__':
    f(1)

