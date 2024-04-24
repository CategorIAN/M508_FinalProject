from Analysis_2 import Analysis_2
from TSP_RL import TSP_RL

class Analysis_3 (Analysis_2):
    def __init__(self):
        super().__init__()
        self.best_hyp_dict = {"p": 4, "T": 2, "eps": 0.01, "n": 3, "alpha": 0.1, "beta": 5, "gamma": 0.9}

    def learnWithBest(self):
        hyp_params = [self.best_hyp_dict[k] for k in self.hyp_names]
        tsp = TSP_RL(*hyp_params)
        G_S_list, Theta = tsp.QLearning(self.graphs)
        distanceTuple = lambda G, S: G.tourDistance(S) / G.distance
        return [distanceTuple(G, S) for G, S in G_S_list]