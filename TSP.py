import numpy as np
from functools import reduce
from Theta import ThetaObject
import random

class TSP:
    def __init__(self, G, T, eps, n, alpha):
        self.G = G
        self.T = T
        self.eps = eps
        self.n = n
        self.N = self.neighbor_matrix()
        self.W = G.dist_matrix.flatten()
        self.U = self.getU()
        self.m = len(self.G.vertices)
        self.alpha = alpha

    def filter(self, s, predicate):
        return reduce(lambda v, i: v + [i] if predicate(i) else v, s, [])

    def c(self, S):
        return - self.G.walkDistance(self.G.closeWalk(S))

    def r(self, S, v):
        return self.c(S+[v]) - self.c(S)

    def x(self, S):
        return np.vectorize(lambda i: int(i in S))(self.G.vertices)

    def relu(self, v):
        return np.vectorize(lambda i: max(0, i))(v)

    def neighbors(self, v):
        return set(self.filter(self.G.vertices, lambda i: i != v))

    def neighbor_vec(self, v):
        neighbors = self.neighbors(v)
        return np.vectorize(lambda i: int(i in neighbors))(self.G.vertices).reshape(-1, 1)

    def neighbor_matrix(self):
        return np.concatenate([self.neighbor_vec(v) for v in self.G.vertices], axis = 1)

    def unit_vec(self, v):
        return np.vectorize(lambda i: int(i == v))(self.G.vertices).reshape(-1, 1)

    def neighbor_square(self, v):
        return np.outer(self.neighbor_vec(v), self.unit_vec(v).reshape(1, -1))

    def getU(self):
        return np.concatenate([self.neighbor_square(v) for v in self.G.vertices], axis=0)

    def F(self, Theta, S):
        x = self.x(S)
        def new_mu_list(mu_list):
            mu = mu_list[-1]
            r1 = np.outer(Theta.theta_1, x)
            r2 = Theta.theta_2 @ mu @ self.N
            r3 = Theta.theta_3 @ self.relu(np.outer(Theta.theta_4, self.W)) @ self.U
            return mu_list + [self.relu(r1 + r2 + r3)]
        return new_mu_list

    def mu_list(self, Theta, S, t_final):
        F_func = self.F(Theta, S)
        mu_recurse = lambda mu_list, t: mu_list if t == t_final else mu_recurse(F_func(mu_list), t + 1)
        return mu_recurse([np.zeros((Theta.p, self.G.n))], 0)

    def mu_list_final(self, Theta, S):
        return self.mu_list(Theta, S, self.T)

    def Q_vec_mu_list(self, Theta, S):
        mu_list = self.mu_list_final(Theta, S)
        mu = mu_list[-1]
        r1 = Theta.theta_5a.reshape(1, -1) @ self.relu(Theta.theta_6 @ mu @ np.ones((self.G.n, self.G.n)))
        r2 = Theta.theta_5b.reshape(1, -1) @ self.relu(Theta.theta_7 @ mu)
        return (r1 + r2)[0], mu_list

    def S_not(self, S):
        return set(self.G.vertices).difference(S)

    def policy(self, Theta, S, S_not = None):
        S_not = self.S_not(S) if S_not is None else S_not
        Qvec, mu_list = self.Q_vec_mu_list(Theta, S)
        vQs = [(v, Qvec[v]) for v in S_not]
        return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, vQs, (None, None)), mu_list

    def vBest(self, Theta, S, S_not = None):
        return self.policy(Theta, S, S_not)[0][0]

    def QBest(self, Theta, S, S_not = None):
        (v, Q) = self.policy(Theta, S, S_not)[0]
        return 0 if Q is None else Q

    def R(self, C, t, n):
        return C[t + n - 1] - C[t - 1]

    def B(self, m, M):
        return random.sample(M, m)

    def updated_S(self, Theta):
        def f(S):
            S_not = self.S_not(S)
            v = np.random.choice(list(S_not)) if np.random.rand() < self.eps else self.vBest(Theta, S, S_not)
            return S + [v]
        return f

    def updated_C(self, S):
        w = self.G.dist_matrix
        def f(C):
            t = len(C)
            c_prev = 0 if len(C) == 0 else C[-1]
            c = c_prev + w[(S[t-1], S[0])] - w[(S[t-1], S[t])] - w[(S[t], S[0])]
            return C + [c]
        return f

    def updated_M(self, S, C, t):
        def f(M):
            return M + [(S[:(t-self.n)], S[t-self.n], self.R(C, t-self.n, self.n), S)] if t >= self.n else M
        return f

    def updated_theta(self, Theta, B):
        def f(i):
            if i in {0, 2, 3, 4, 5, 6, 7}:
                batch_gradient = 1 / len(B) * sum([self.dJ_dtheta(Theta, *b, i).T for b in B])
                return Theta.thetas[i] - self.alpha * batch_gradient
            else:
                return Theta.thetas[i]
        return f

    def updated_Theta(self, M, t):
        def f(Theta):
            if t >= self.n:
                r = min(len(M), 5)
                B = random.sample(M, r)
                theta_func = self.updated_theta(Theta, B)
                thetas_new = [theta_func(i) for i in range(8)]
                return ThetaObject(thetas_new)
            else:
                return Theta
        return f

    def QLearning(self, Theta):
        def QLearn_recurse(S, C, M, Theta, t):
            print(50 * "=")
            print("S: {}".format(S))
            print("C: {}".format(C))
            print("M: {}".format(M))
            print("Theta:\n{}".format(Theta))
            print("t: {}".format(t))
            if t == self.m:
                return Theta
            else:
                S_new = self.updated_S(Theta)(S)
                C_new = self.updated_C(S_new)(C)
                t = len(S)
                M_new = self.updated_M(S, C, t)(M)
                Theta_new = self.updated_Theta(M_new, t)(Theta)
                return QLearn_recurse(S_new, C_new, M_new, Theta_new, t + 1)
        S, C, M = ([], [], [])
        return QLearn_recurse(S, C, M, Theta, 0)

    def dJ_dtheta(self, Theta, S_past, v_past, R_diff, S, i):
        z = R_diff + self.QBest(Theta, S) - self.Q_vec_mu_list(Theta, S_past)[0][v_past]
        return z * self.dz_dtheta(Theta, S_past, v_past, S, i)

    def dz_dtheta(self, Theta, S_past, v_past, S, i):
        return self.dQBest_dtheta(Theta, S, i) - self.dQ_dtheta(Theta, S_past, v_past, i)

    def dQ_dtheta(self, Theta, S, v, i):
        Q_vec, mu_list = self.Q_vec_mu_list(Theta, S)
        ra_1 = Theta.theta_6 @ mu_list[-1] @ np.ones((self.m, 1))
        rb_1 = Theta.theta_7 @ mu_list[-1] @ self.unit_vec(v)
        if i in {0, 2, 3}:
            ra_2 = Theta.theta_5a.reshape(1, -1) @ self.dRelu_dz(ra_1) @ Theta.theta_6
            ra = sum([self.dmu_dtheta(ra_2, Theta, S, i, mu_list[:-1], u) for u in self.G.vertices])
            rb_2 = Theta.theta_5b.reshape(1, -1) @ self.dRelu_dz(rb_1) @ Theta.theta_7
            rb = self.dmu_dtheta(rb_2, Theta, S, i, mu_list[:-1], v)
            return ra + rb
        elif i == 4:
            return self.relu(ra_1).reshape(1, -1)
        elif i == 5:
            return self.relu(rb_1).reshape(1, -1)
        elif i == 6:
            return np.outer(mu_list[-1] @ np.ones((self.m, 1)), Theta.theta_5a.reshape(1, -1) @ self.dRelu_dz(ra_1))
        elif i == 7:
            return np.outer(mu_list[-1] @ self.unit_vec(v), Theta.theta_5b.reshape(1, -1) @ self.dRelu_dz(rb_1))

    def dQBest_dtheta(self, Theta, S, i):
        S_not = self.S_not(S)
        if len(S_not) == 0:
            return 0
        else:
            (v, Q), mu_list = self.policy(Theta, S, S_not)
            return self.dQ_dtheta(Theta, S, v, i)

    def dRelu_dz(self, z):
        return np.diag(np.vectorize(lambda i: int(i > 0))(z)[:, 0])

    def dmu_dtheta(self, r, Theta, S, i, mu_list, v):
        if len(mu_list) == 0:
            if i in {0, 3}:
                return np.zeros((1, Theta.p))
            elif i in {1, 2}:
                return np.zeros((Theta.p, Theta.p))
        else:
            z = (Theta.theta_1 * self.x(S)[v] + Theta.theta_2 @ mu_list[-1] @ self.N[:, v] +
            Theta.theta_3 @ self.relu(np.outer(Theta.theta_4, self.W)) @ self.U[:, v])
            r_new = r @ self.dRelu_dz(z)
            r_1 = self.dthetafunc_dtheta_1(r_new, Theta, S, i, v)
            r_2 = self.dthetafunc_dtheta_2(r_new, Theta, S, i, mu_list, v)
            r_3 = self.dthetafunc_dtheta_3(r_new, Theta, i, v)
            return r_1 + r_2 + r_3

    def dthetafunc_dtheta_1(self, r, Theta, S, i, v):
        if i in {0}:
            return r @ (self.x(S)[v] * np.eye(Theta.p))
        elif i in {2}:
            return np.zeros((Theta.p, Theta.p))
        elif i in {3}:
            return np.zeros((1, Theta.p))

    def dthetafunc_dtheta_2(self, r, Theta, S, i, mu_list, v):
        if i in {0, 2, 3}:
            r_new = r @ Theta.theta_2
            return sum([self.dmu_dtheta(r_new, Theta, S, i, mu_list[:-1], u) for u in self.neighbors(v)])
        else:
            return np.zeros((1, Theta.p))

    def dthetafunc_dtheta_3(self, r, Theta, i, v):
        if i in {2}:
            return np.outer(self.relu(np.outer(Theta.theta_4, self.W)) @ self.U @ self.unit_vec(v), r)
        elif i in {3}:
            w = self.G.dist_matrix
            return (r @ Theta.theta_3 @ sum([self.dRelu_dz(Theta.theta_4 * w[(v, u)]) @
                                         (w[(v, u)] * np.eye(Theta.p, Theta.p)) for u in self.neighbors(v)]))
        else:
            return np.zeros((1, Theta.p))











