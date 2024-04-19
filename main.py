from TSP import TSP

def f(i):
    if i == 1:
        p, T, eps, n, alpha, beta, gamma = 3, 2, 0.01, 1, 0.01, 5, 0.9
        tsp = TSP(p, T, eps, n, alpha, beta, gamma)
        G_S_list, (Theta_final, M_final) = tsp.QLearning(5)
        print(G_S_list)

if __name__ == '__main__':
    f(1)

