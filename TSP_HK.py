from itertools import combinations
from functools import reduce

class TSP_HK:
    def policy(self, G, P, v):
        m = len(P)
        if m == 1:
            return G.w[(v, 0)]
        else:
            Q = P - {v}
            vDs = [(u, G.w[(v, u)] + self.vBest(Q, u)) for u in Q]
            return reduce(lambda t1, t2: t2 if t1[0] is None or t2[1] > t1[1] else t1, vDs, (None, 0))
        pass

    def vBest(self, G, P, v):
        return self.policy(G, P, v)[0]

    def QBest(self, G, P, v):
        return self.policy(G, P, v)[1]

    def subPaths(self, P, v):
        pass

