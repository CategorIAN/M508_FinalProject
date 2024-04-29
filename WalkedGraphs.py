from EuclideanGraph import EuclideanGraph
import pandas as pd

class WalkedGraphs:
    def __init__(self, file, graphDist = None, n = None):
        self.graphs = self.createGraphs(graphDist, n) if file is None else self.getGraphsFromCSV(file)

    def getGraphsFromCSV(self, file):
        df = pd.read_csv(file, index_col=0)
        m = len(df["Graph"].unique())
        def graph(i):
            graph_df = df.loc[lambda df: df["Graph"] == i]
            ordered_points = list(zip(list(graph_df["x"]), list(graph_df["y"])))
            S = list(range(len(ordered_points)))
            d = graph_df["d"].iloc[0]
            return EuclideanGraph(points=ordered_points, shortest_cycle=S, distance=d)
        return [graph(i) for i in range(m)]

    def createGraphs(self, graphDist, n):
        return [graphDist.randomGraph().thisWithShortestCycle() for i in range(n)]

    def toCSV(self, graphs, file):
        def graphDF(i):
            G = graphs[i]
            ordered_points = [G.points[j] for j in G.shortest_cycle]
            d = G.distance
            xs, ys = tuple(zip(*ordered_points))
            return pd.DataFrame(data = {"Graph": G.n * [i], "x": xs, "y": ys, "d(G)": G.n * [d]})
        df = pd.concat([graphDF(i) for i in range(len(graphs))], axis=0).reset_index(drop=True)
        df.to_csv(file)
        return df

    def getGraphs(self, file, graphDist, n):
        try:
            return self.getGraphsFromCSV(file)
        except:
            graphs = self.createGraphs(graphDist, n)
            self.toCSV(graphs, file)
            return self.getGraphsFromCSV(file)








