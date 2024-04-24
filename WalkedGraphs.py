from RandomEuclideanGraph import RandomEuclideanGraph
from EuclideanGraph import EuclideanGraph
import pandas as pd
import os

class WalkedGraphs:
    def __init__(self, n = None, csv = "WalkedGraphs.csv"):
        if csv is not None:
            self.graphs = self.getGraphsFromCSV(csv)
            self.n = len(self.graphs)
        else:
            self.n = n
            self.graphs = self.createGraphs()

    def createGraphs(self):
        def walkedGraph():
            G = RandomEuclideanGraph()
            return G.thisWithShortestCycle()
        return [walkedGraph() for i in range(self.n)]

    def getCSV(self, csv = "WalkedGraphs.csv"):
        def graphDF(i):
            G = self.graphs[i]
            ordered_points = [G.points[j] for j in G.shortest_cycle]
            d = G.distance
            xs, ys = tuple(zip(*ordered_points))
            return pd.DataFrame(data = {"Graph": G.n * [i], "x": xs, "y": ys, "d": G.n * [d]})
        df = pd.concat([graphDF(i) for i in range(self.n)], axis=0).reset_index(drop=True)
        df.to_csv("\\".join([os.getcwd(), csv]))
        return df

    def getGraphsFromCSV(self, csv):
        df = pd.read_csv("\\".join([os.getcwd(), csv]), index_col=0)
        m = len(df["Graph"].unique())
        def graph(i):
            graph_df = df.loc[lambda df: df["Graph"] == i]
            ordered_points = list(zip(list(graph_df["x"]), list(graph_df["y"])))
            S = list(range(len(ordered_points)))
            d = graph_df["d"].iloc[0]
            return EuclideanGraph(points=ordered_points, shortest_cycle=S, distance=d)
        return [graph(i) for i in range(m)]





