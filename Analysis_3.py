from Analysis_2 import Analysis_2
from TSP_RL import TSP_RL
import pandas as pd
from EuclideanGraph import EuclideanGraph
import matplotlib.pyplot as plt
from WalkedGraphs import WalkedGraphs

class Analysis_3 (Analysis_2):
    def __init__(self, graph_file = None, graphDist = None, n = None):
        super().__init__(graph_file, graphDist, n)
        self.best_hyp_dict = {"p": 4, "T": 2, "eps": 0.01, "n": 3, "alpha": 0.1, "beta": 5, "gamma": 0.9}
        self.best_hyp_params = [self.best_hyp_dict[k] for k in self.hyp_names]

    def inversePerm(self, S):
        index = list(range(len(S)))
        inverseDict = dict(list(zip(S, index)))
        return [inverseDict[i] for i in index]

    def newFile(self, file, note, ext = ".csv"):
        file_path = file.split("\\")
        csv_file = file_path[-1].strip(".csv")
        return "\\".join(file_path[:-1] + ["{} ({})".format(csv_file, note) + ext])

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

    def toCSV(self, graph_file, graphDist = None, n = None):
        graphs = WalkedGraphs(graph_file, graphDist, n).graphs
        tsp = TSP_RL(*self.best_hyp_params)
        G_S_list, _ = tsp.QLearning(graphs)
        graphDF_func = self.graphDF(G_S_list)
        df = pd.concat([graphDF_func(i) for i in range(len(G_S_list))], axis=0).reset_index(drop=True)
        output_file = self.newFile(graph_file, "approxS")
        df.to_csv(output_file)
        return df

    def getG_S_rFromCSV(self, file):
        df = pd.read_csv(file, index_col=0)
        m = len(df["Graph"].unique())
        def G_S_r(i):
            graph_df = df.loc[lambda df: df["Graph"] == i]
            ordered_points = list(zip(list(graph_df["x"]), list(graph_df["y"])))
            S_true = list(range(len(ordered_points)))
            d_true = graph_df["actualDist"].iloc[0]
            S_approx = self.inversePerm(graph_df["approxOrder"])
            approx_ratio = graph_df["approxRatio"].iloc[0]
            return EuclideanGraph(points=ordered_points, shortest_cycle=S_true, distance=d_true), S_approx, approx_ratio
        return [G_S_r(i) for i in range(m)]

    def getG_S_r(self, graph_file, graphDist, n):
        try:
            return self.getG_S_rFromCSV(self.newFile(graph_file, "approxS"))
        except FileNotFoundError:
            self.toCSV(graph_file, graphDist, n)
            return self.getG_S_rFromCSV(self.newFile(graph_file, "approxS"))

    def drawGraphs(self, graph_file, graphDist, n):
        G_S_r_list = self.getG_S_r(graph_file, graphDist, n)

        def graph_plot(i):
            G, S, r = G_S_r_list[i]
            ordered_points = G.points
            closed = lambda z: z + (z[0],)
            x_true, y_true = tuple(zip(*ordered_points))
            plt.suptitle("Learned Path of Graph {} After Episode {} (Approx Ratio: {})".format(i, i, round(r, 2)))
            plt.plot(x_true, y_true, "o")
            x_approx, y_approx = tuple(zip(*[G.points[j] for j in S]))
            plt.plot(closed(x_approx), closed(y_approx), "-", label="Approximated")
            plt.plot(closed(x_true), closed(y_true), "-", label = "Optimal")
            plt.legend()
            plt.savefig(self.newFile(graph_file, "Graph {} Plot".format(i), ext=".png"))
            plt.show()
        [graph_plot(i) for i in range(len(G_S_r_list))]
        episodes, ratios = tuple(zip(*[(i, G_S_r_list[i][2]) for i in range(len(G_S_r_list))]))
        plt.suptitle("Approx Ratio vs. Episode")
        plt.xlabel("Episode")
        plt.ylabel("Approx Ratio")
        plt.bar(episodes, ratios)
        plt.savefig(self.newFile(graph_file, "Approx Ratio vs. Episode", ext=".png"))
        plt.show()



