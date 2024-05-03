from EuclideanGraph import EuclideanGraph
import pandas as pd

class WalkedGraphs:
    def __init__(self, file, graphDist = None, n = None):
        '''
        :param file: the file to read from or to write to for the walked graphs
        :param graphDist: the graph distribution used to generate graphs (if the file is not already created)
        :param n: the number of graphs to generate (if the file is not already created)
        '''
        self.graphs = self.getGraphs(file, graphDist, n)

    def getGraphsFromCSV(self, file):
        '''
        :param file: the file to get the graphs from
        :return: list of graphs that have a calculated permutation S and distance d best for TSP
        '''
        df = pd.read_csv(file, index_col=0)
        m = len(df["Graph"].unique())
        def graph(i):
            graph_df = df.loc[lambda df: df["Graph"] == i]
            ordered_points = list(zip(list(graph_df["x"]), list(graph_df["y"])))
            S = list(range(len(ordered_points)))
            d = graph_df["d(G)"].iloc[0]
            return EuclideanGraph(points=ordered_points, shortest_cycle=S, distance=d)
        return [graph(i) for i in range(m)]

    def createGraphs(self, graphDist, n):
        '''
        :param graphDist: graph distribution to generate graphs
        :param n: the number of graphs to generate
        :return: list of graphs that have a calculated permutation S and distance d best for TSP
        '''
        return [graphDist.randomGraph().thisWithShortestCycle() for i in range(n)]

    def toCSV(self, graphs, file):
        '''
        :param graphs: list of graphs that have a calculated permutation S and distance d best for TSP
        :param file: file to store the graphs
        :return: the dataframe used to store graphs (side effect: saved this dataframe to the file)
        '''
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
        '''
        :param file: the file to get the graphs from
        :param graphDist: the graph distribution used to generate graphs (if the file is not already created)
        :param n: the number of graphs to generate (if the file is not already created)
        :return: list of graphs that have a calculated permutation S and distance d best for TSP
        '''
        try:
            return self.getGraphsFromCSV(file)
        except FileNotFoundError:
            graphs = self.createGraphs(graphDist, n)
            self.toCSV(graphs, file)
            return self.getGraphsFromCSV(file)








