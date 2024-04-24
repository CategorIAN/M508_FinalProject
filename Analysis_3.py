from Analysis_2 import Analysis_2
from TSP_RL import TSP_RL
import pandas as pd
import os

class Analysis_3 (Analysis_2):
    def __init__(self):
        super().__init__()
        self.best_hyp_dict = {"p": 4, "T": 2, "eps": 0.01, "n": 3, "alpha": 0.1, "beta": 5, "gamma": 0.9}

    def inversePerm(self, S):
        index = list(range(len(S)))
        inverseDict = dict(list(zip(S, index)))
        return [inverseDict[i] for i in index]

    def graphDF(self, G_S_list):
        def f(i):
            G, S = G_S_list[i]
            ordered_points = [G.points[j] for j in G.shortest_cycle]
            xs, ys = tuple(zip(*ordered_points))
            actualDist = G.distance
            approxOrder = self.inversePerm(S)
            approxDist = G.tourDistance(S)
            approxRatio = approxDist / actualDist
            return pd.DataFrame(data={"Graph": G.n * [i], "x": xs, "y": ys, "approxOrder": approxOrder,
                "actualDist": G.n * [actualDist], "approxDist": G.n * [approxDist], "approxRatio": G.n * [approxRatio]})
        return f

    def getG_S_CSV(self, csv = "WalkedGraphsWithApproxS.csv"):
        hyp_params = [self.best_hyp_dict[k] for k in self.hyp_names]
        tsp = TSP_RL(*hyp_params)
        G_S_list, Theta = tsp.QLearning(self.graphs)
        graphDF_func = self.graphDF(G_S_list)
        df = pd.concat([graphDF_func(i) for i in range(len(G_S_list))], axis=0).reset_index(drop=True)
        df.to_csv("\\".join([os.getcwd(), csv]))
        return df