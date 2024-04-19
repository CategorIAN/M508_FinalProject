import numpy as np
from functools import reduce
from Theta import ThetaObject
import random
from RandomEuclideanGraph import RandomEuclideanGraph
from Theta import RandomThetaObject

class TSP:
    def __init__(self, p, T, eps, n, alpha, beta, gamma):
        '''
        :param p: the dimension of the vertex embedding
        :param T: the number of hidden layers
        :param eps: the probability of choosing a random vertex
        :param n: the number of steps to go before Q learning
        :param alpha: the learning rate of gradient descent
        :param beta: the maximum size of batches for batch gradient descent
        :param gamma: the discount factor
        '''
        self.p = p
        self.T = T
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def filter(self, s, predicate):
        '''
        :param s: list of values
        :param predicate: function to determine what to keep in the list
        :return: sublist of values that satisfied the predicate
        '''
        return reduce(lambda v, i: v + [i] if predicate(i) else v, s, [])

    def c(self, G, S):
        '''
        :param G: graph
        :param S: list of unique vertices of graph
        :return: negative total cost to travel the tour generated from the list of vertices
        '''
        return - G.walkDistance(G.closeWalk(S))

    def r(self, G, S, v):
        '''
        :param G: graph
        :param S: list of unique vertices of graph
        :param v: vertex to append to list
        :return: the change in cost from the tour of S to the tour of S + [v]
        '''
        return self.c(G, S+[v]) - self.c(G, S)

    def x(self, G, S):
        '''
        :param G: graph
        :param S: list of unique vertices of graph
        :return: vector of binary values showing which vertices in G are in S
        '''
        return np.vectorize(lambda i: int(i in S))(G.vertices)

    def relu(self, v):
        '''
        :param v: vector
        :return: vector showing the element-wise relu values of v
        '''
        return np.vectorize(lambda i: max(0, i))(v)

    def F(self, Theta, G, S):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :return: function that takes a list of mu's and returns that list with an appended next mu
        '''
        x = self.x(G, S)
        def new_mu_list(mu_list):
            mu = mu_list[-1]
            r1 = np.outer(Theta.theta_1, x)
            r2 = Theta.theta_2 @ mu @ G.N
            r3 = Theta.theta_3 @ self.relu(np.outer(Theta.theta_4, G.W)) @ G.U
            return mu_list + [self.relu(r1 + r2 + r3)]
        return new_mu_list

    def mu_list_final(self, Theta, G, S):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :return: final list of mu matrices
        '''
        F_func = self.F(Theta, G, S)
        mu_recurse = lambda mu_list, t: mu_list if t == self.T else mu_recurse(F_func(mu_list), t + 1)
        return mu_recurse([np.zeros((Theta.p, G.n))], 0)

    def Q_vec_mu_list(self, Theta, G, S):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :return: a tuple of the Q vector for S and the mu list used to calculate Q vector
        '''
        mu_list = self.mu_list_final(Theta, G, S)
        mu = mu_list[-1]
        r1 = Theta.theta_5a.reshape(1, -1) @ self.relu(Theta.theta_6 @ mu @ np.ones((G.n, G.n)))
        r2 = Theta.theta_5b.reshape(1, -1) @ self.relu(Theta.theta_7 @ mu)
        return (r1 + r2)[0], mu_list

    def S_not(self, G, S):
        '''
        :param G: graph
        :param S: list of unique vertices of graph
        :return: set of vertices in G but not in S
        '''
        return set(G.vertices).difference(S)

    def policy(self, Theta, G, S, S_not = None):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :param S_not: (optional) set of vertices in G but not in S (to speed up calculation)
        :return: ((v, Q), mu_list) where (v, Q) is the (argmax, max) of the Q vector and mu_list was used for Q vector
        '''
        S_not = self.S_not(G, S) if S_not is None else S_not
        Qvec, mu_list = self.Q_vec_mu_list(Theta, G, S)
        vQs = [(v, Qvec[v]) for v in S_not]
        return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, vQs, (None, 0)), mu_list

    def vBest(self, Theta, G, S, S_not = None):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :param S_not: (optional) set of vertices in G but not in S (to speed up calculation)
        :return: the best vertex to append to S if S_not is nonempty else None
        '''
        return self.policy(Theta, G, S, S_not)[0][0]

    def QBest(self, Theta, G, S, S_not = None):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :param S_not: (optional) set of vertices in G but not in S (to speed up calculation)
        :return: Q value of the best vertex to append to S if S_not is nonempty else 0
        '''
        return self.policy(Theta, G, S, S_not)[0][1]

    def R(self, C, t, n):
        '''
        :param C: vector of accumulated costs
        :param t: current time index
        :param n: number of steps into the future
        :return: change in rewards from current time to n steps in the future
        '''
        return C[t + n - 1] - C[t - 1]

    def dRelu_dz(self, z):
        '''
        :param z: argument vector of relu function
        :return: matrix that is the derivative of relu(z) with respect to z
        '''
        return np.diag(np.vectorize(lambda i: int(i > 0))(z)[:, 0])

    def dthetafunc_dtheta_1(self, r, Theta, G, S, i, v):
        '''
        :param r: horizontal vector that multiplies to derivative numerator
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :param i: index of Theta list for the theta that is used for the derivative denominator
        :param v: vector column used in mu expression
        :return: vector or matrix representing a derivative
        '''
        if i in {0}:
            return r @ (self.x(G, S)[v] * np.eye(Theta.p))
        elif i in {1, 2}:
            return np.zeros((Theta.p, Theta.p))
        elif i in {3}:
            return np.zeros((1, Theta.p))

    def dthetafunc_dtheta_2(self, r, Theta, G, S, i, mu_list, v):
        '''
        :param r: horizontal vector that multiplies to derivative numerator
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :param i: index of Theta list for the theta that is used for the derivative denominator
        :param mu_list: list of mu matrices where its last entry is the mu used in expression
        :param v: vector column used in mu expression
        :return: vector or matrix representing a derivative
        '''
        r_new = r @ Theta.theta_2
        neighbor_dmu = lambda u: self.dmu_dtheta(r_new, Theta, G, S, i, mu_list[:-1], u)
        if i in {0, 2, 3}:
            return sum([neighbor_dmu(u) for u in G.neighbors[v]])
        elif i in {1}:
            neighbor_outerproduct = lambda u: np.outer(mu_list[-1] @ G.unit_vec(u), r)
            return sum([neighbor_dmu(u) + neighbor_outerproduct(u) for u in G.neighbors[v]])

    def dthetafunc_dtheta_3(self, r, Theta, G, i, v):
        '''
        :param r: horizontal vector that multiplies to derivative numerator
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param i: index of Theta list for the theta that is used for the derivative denominator
        :param v: vector column used in mu expression
        :return: vector or matrix representing a derivative
        '''
        if i in {0}:
            return np.zeros((1, Theta.p))
        elif i in {1}:
            return np.zeros((Theta.p, Theta.p))
        elif i in {2}:
            return np.outer(self.relu(np.outer(Theta.theta_4, G.W)) @ G.U @ G.unit_vec(v), r)
        elif i in {3}:
            return (r @ Theta.theta_3 @ sum([self.dRelu_dz(Theta.theta_4 * G.w[(v, u)]) @
                                         (G.w[(v, u)] * np.eye(Theta.p, Theta.p)) for u in G.neighbors[v]]))

    def dmu_dtheta(self, r, Theta, G, S, i, mu_list, v):
        '''
        :param r: horizontal vector that multiplies to derivative numerator
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :param i: index of Theta list for the theta that is used for the derivative denominator
        :param mu_list: list of mu matrices where its last entry is the mu to calculate the current mu
        :param v: vector column used in mu expression
        :return: vector or matrix representing a derivative
        '''
        if len(mu_list) == 0:
            if i in {0, 3}:
                return np.zeros((1, Theta.p))
            elif i in {1, 2}:
                return np.zeros((Theta.p, Theta.p))
        else:
            z = (Theta.theta_1 * self.x(G, S)[v] + Theta.theta_2 @ mu_list[-1] @ G.N[:, v] +
            Theta.theta_3 @ self.relu(np.outer(Theta.theta_4, G.W)) @ G.U[:, v])
            r_new = r @ self.dRelu_dz(z)
            r_1 = self.dthetafunc_dtheta_1(r_new, Theta, G, S, i, v)
            r_2 = self.dthetafunc_dtheta_2(r_new, Theta, G, S, i, mu_list, v)
            r_3 = self.dthetafunc_dtheta_3(r_new, Theta, G, i, v)
            return r_1 + r_2 + r_3

    def dQ_dtheta(self, Theta, G, S, v, i):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :param i: index of Theta list for the theta that is used for the derivative denominator
        :param v: column used in Q vector
        :return: vector or matrix representing a derivative
        '''
        Q_vec, mu_list = self.Q_vec_mu_list(Theta, G, S)
        ra_1 = Theta.theta_6 @ mu_list[-1] @ np.ones((G.n, 1))
        rb_1 = Theta.theta_7 @ mu_list[-1] @ G.unit_vec(v)
        if i in {0, 1, 2, 3}:
            ra_2 = Theta.theta_5a.reshape(1, -1) @ self.dRelu_dz(ra_1) @ Theta.theta_6
            ra = sum([self.dmu_dtheta(ra_2, Theta, G, S, i, mu_list[:-1], u) for u in G.vertices])
            rb_2 = Theta.theta_5b.reshape(1, -1) @ self.dRelu_dz(rb_1) @ Theta.theta_7
            rb = self.dmu_dtheta(rb_2, Theta, G, S, i, mu_list[:-1], v)
            return ra + rb
        elif i == 4:
            return self.relu(ra_1).reshape(1, -1)
        elif i == 5:
            return self.relu(rb_1).reshape(1, -1)
        elif i == 6:
            return np.outer(mu_list[-1] @ np.ones((G.n, 1)), Theta.theta_5a.reshape(1, -1) @ self.dRelu_dz(ra_1))
        elif i == 7:
            return np.outer(mu_list[-1] @ G.unit_vec(v), Theta.theta_5b.reshape(1, -1) @ self.dRelu_dz(rb_1))

    def dQBest_dtheta(self, Theta, G, S, i):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S: list of unique vertices of graph
        :param i: index of Theta list for the theta that is used for the derivative denominator
        :return: vector or matrix representing a derivative
        '''
        S_not = self.S_not(G, S)
        if len(S_not) == 0:
            return 0
        else:
            (v, Q), mu_list = self.policy(Theta, G, S, S_not)
            return self.dQ_dtheta(Theta, G, S, v, i)

    def dz_dtheta(self, Theta, G, S_past, v_past, S, i):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S_past: list of unique vertices of graph n steps in the past
        :param v_past: vertex appended to S_past n steps in the past
        :param S: current list of unique vertices of graph
        :param i: index of Theta list for the theta that is used for the derivative denominator
        :return: vector or matrix representing a derivative
        '''
        return self.gamma * self.dQBest_dtheta(Theta, G, S, i) - self.dQ_dtheta(Theta, G, S_past, v_past, i)

    def dJ_dtheta(self, Theta, G, S_past, v_past, R_diff, S, i):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :param S_past: list of unique vertices of graph n steps in the past
        :param v_past: vertex appended to S_past n steps in the past
        :param R_diff: change in rewards from the past to the present
        :param S: current list of unique vertices of graph
        :param i: index of Theta list for the theta that is used for the derivative denominator
        :return: vector or matrix representing a derivative
        '''
        z = R_diff + self.gamma * self.QBest(Theta, G, S) - self.Q_vec_mu_list(Theta, G, S_past)[0][v_past]
        return z * self.dz_dtheta(Theta, G, S_past, v_past, S, i)

    def updated_S(self, Theta, G):
        '''
        :param Theta: object consisting of list of theta weights
        :param G: graph
        :return: function that updates vertex list S
        '''
        def f(S):
            S_not = self.S_not(G, S)
            v = np.random.choice(list(S_not)) if np.random.rand() < self.eps else self.vBest(Theta, G, S, S_not)
            return S + [v]
        return f

    def updated_C(self, G, S):
        '''
        :param G: graph
        :param S: list of unique vertices of graph
        :return: function that updates cost list C
        '''
        def f(C):
            t = len(C)
            c_prev = 0 if len(C) == 0 else C[-1]
            c = c_prev + G.w[(S[t-1], S[0])] - G.w[(S[t-1], S[t])] - G.w[(S[t], S[0])]
            return C + [c]
        return f

    def updated_M(self, G, S, C, t):
        '''
        :param G: graph
        :param S: list of unique vertices of graph
        :param C: list of accumulated costs S
        :param t: current time index
        :return: function that updates memory list M
        '''
        def f(M):
            return M + [(G, S[:(t-self.n)], S[t-self.n], self.R(C, t-self.n, self.n), S)] if t >= self.n else M
        return f

    def updated_theta(self, Theta, B):
        '''
        :param Theta: object consisting of list of theta weights
        :param B: random batch from memory list
        :return: function that takes theta index and returns updated theta for that index
        '''
        def f(i):
            if i in {0, 1, 2, 3, 4, 5, 6, 7}:
                batch_gradient = 1 / len(B) * sum([self.dJ_dtheta(Theta, *b, i).T for b in B])
                return Theta.thetas[i] - self.alpha * batch_gradient
            else:
                return Theta.thetas[i]
        return f

    def updated_Theta(self, M, t):
        '''
        :param M: memory list
        :param t: current time index
        :return: function that updates object Theta
        '''
        def f(Theta):
            if t >= self.n:
                r = min(len(M), self.beta)
                B = random.sample(M, r)
                theta_func = self.updated_theta(Theta, B)
                thetas_new = [theta_func(i) for i in range(8)]
                return ThetaObject(thetas_new)
            else:
                return Theta
        return f

    def episode(self, Theta, M, G):
        '''
        :param Theta: object consisting of list of theta weights
        :param M: memory list
        :param G: graph
        :return: (Theta_new, M_new, S) where (Theta_new, M_new) are updated and S is the permutation used for graph G
        '''
        print(100 * "#")
        print(G)
        def QLearn_recurse(S, C, M, Theta, t):
            """
            print(50 * "=")
            print("S: {}".format(S))
            print("C: {}".format(C))
            print("M: {}".format(M))
            print("Theta:\n{}".format(Theta))
            print("t: {}".format(t))
            """
            if t == G.n:
                return Theta, M, S
            else:
                S_new = self.updated_S(Theta, G)(S)
                C_new = self.updated_C(G, S_new)(C)
                t = len(S)
                M_new = self.updated_M(G, S, C, t)(M)
                Theta_new = self.updated_Theta(M_new, t)(Theta)
                return QLearn_recurse(S_new, C_new, M_new, Theta_new, t + 1)

        return QLearn_recurse([], [], M, Theta, 0)

    def QLearning(self, L):
        '''
        :param L: number of episodes to use
        :return: (G_S_list, (Theta, M)) where G_S_list is a list of (G, S) for graph G and its corresponding
                 permutation S to use and (Theta, M) are updated values from the Q learning
        '''
        def appendEpisodeResults(results, G):
            G_S_list, (Theta, M) = results
            Theta_new, M_new, S = self.episode(Theta, M, G)
            return G_S_list + [(G, S)], (Theta_new, M_new)

        Gs = [RandomEuclideanGraph() for i in range(L)]
        return reduce(appendEpisodeResults, Gs, ([], (RandomThetaObject(self.p), [])))












