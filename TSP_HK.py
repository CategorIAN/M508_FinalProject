from itertools import combinations
from functools import reduce

class TSP_HK:
    def __init__(self):
        pass

    def policy(self, G, Q_dict):
        '''
        :param G: graph
        :param Q_dict: tabular dynamic dictionary
        :return: function that takes path P and starting vertex c and returns (v, Q) where v is next best vertex to
        travel to and Q is the dynamic value of (P, c)
        '''
        def f(P, c):
            R = P - {c}
            vQs = [(u, G.w[(c, u)] + Q_dict[(R, u)]) for u in R]
            return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] < t1[1] else t1, vQs, (None, G.w[(c, 0)]))
        return f

    def vBest(self, G, Q_dict):
        '''
        :param G: graph
        :param Q_dict: tabular dynamic dictionary
        :return: function that takes path P and starting vertex c and returns the next best vertex to travel to
        '''
        return lambda P, c: self.policy(G, Q_dict)(P, c)[0]

    def QBest(self, G, Q_dict):
        '''
        :param G: graph
        :param Q_dict: tabular dynamic dictionary
        :return: function that takes path P and starting vertex c and returns the dynamic value of (P, c)
        '''
        return lambda P, c: self.policy(G, Q_dict)(P, c)[1]

    def getQfromP(self, G, Q_dict):
        '''
        :param G: graph
        :param Q_dict: tabular dynamic dictionary
        :return: function that takes path P and returns dictionary for keys that have (P, _)
        '''
        Q_func = self.QBest(G, Q_dict)
        return lambda P: dict([((P, c), Q_func(P, c)) for c in P])

    def createQdict(self, G):
        '''
        :param G: graph
        :return: final dynamic dictionary of subproblems built from the bottom up
        '''
        #print("Creating Q Dict")
        neighbors = [v for v in G.vertices if v != 0]
        def createLevel(Q_dict, i):
            #print("Creating Level {}".format(i))
            if i == G.n:
                return Q_dict
            else:
                X = [frozenset(x) for x in combinations(neighbors, i)]
                Q_dict_func = self.getQfromP(G, Q_dict)
                Q_dict_new = reduce(lambda Q, P: Q | Q_dict_func(P), X, Q_dict)
                return createLevel(Q_dict_new, i + 1)
        return createLevel({}, 1)

    def S_not(self, G, S):
        '''
        :param G: graph
        :param S: list of unique vertices of graph
        :return: set of vertices in G but not in S
        '''
        return set(G.vertices).difference(S)

    def calculateWalkfromQ(self, G, Q_dict):
        '''
        :param G: graph
        :param Q_dict: dynamic dictionary that contains the subproblems
        :return: permutation of vertices of G that give the shortest Hamiltonian cycle
        '''
        #print("Calculating Walk From Q")
        def updated_S(i, S):
            if i == G.n:
                return S
            else:
                S_not = self.S_not(G, S)
                c = S[-1]
                P = frozenset(S_not).union({c})
                v = self.vBest(G, Q_dict)(P, c)
                return updated_S(i + 1, S + [v])
        return updated_S(1, [0])

    def calculateWalk(self, G):
        '''
        :param G: graph
        :return: permutation of vertices of G that give the shortest Hamiltonian cycle
        '''
        Q_dict = self.createQdict(G)
        return self.calculateWalkfromQ(G, Q_dict)








