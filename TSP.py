

class TSP:
    def __init__(self, G):
        self.G = G
    def h(self, S):
        return S
    def c(self, H):
        return self.G.walkDistance(self.G.closeWalk(H))