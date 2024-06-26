from Analysis_2 import Analysis_2
from TSP_RL import TSP_RL
import pandas as pd
from EuclideanGraph import EuclideanGraph
import matplotlib.pyplot as plt
from WalkedGraphs import WalkedGraphs

class Analysis_3 (Analysis_2):
    def __init__(self, graph_file = None, graphDist = None, n = None):
        '''
        :param graph_file: the file to read from or to write to for the walked graphs
        :param graphDist: the graph distribution used to generate graphs (if the file is not already created)
        :param n: the number of graphs to generate (if the file is not already created)
        '''
        super().__init__(graph_file, graphDist, n)
        self.best_hyp_dict = {"p": 4, "T": 2, "eps": 0.01, "n": 3, "alpha": 0.1, "beta": 5, "gamma": 0.9}
        self.best_hyp_params = [self.best_hyp_dict[k] for k in self.hyp_names]

    def inversePerm(self, S):
        '''
        :param S: permutation
        :return: the inverse permutation of S
        '''
        index = list(range(len(S)))
        inverseDict = dict(list(zip(S, index)))
        return [inverseDict[i] for i in index]

    def newFile(self, file, note, ext = ".csv"):
        '''
        :param file: file to create a new file from
        :param note: additional note used to name new file
        :param ext: extension of file
        :return: new file
        '''
        file_path = file.split("\\")
        csv_file = file_path[-1].strip(".csv")
        return "\\".join(file_path[:-1] + ["{} ({})".format(csv_file, note) + ext])

    def graphDF(self, G_S_list):
        '''
        :param G_S_list: list of (G, S) where G is a graph and S is permutation to approximate TSP
        :return: dataframe that stores the graph information as well as S and walk distance that approximate TSP
        '''
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
        '''
        :param graph_file: the file to read from or to write to for the walked graphs
        :param graphDist: the graph distribution used to generate graphs (if the file is not already created)
        :param n: the number of graphs to generate (if the file is not already created)
        :return: dataframe of graphs with approximated best S and walk distance (side effect: stores this in a new file)
        '''
        graphs = WalkedGraphs(graph_file, graphDist, n).graphs
        tsp = TSP_RL(*self.best_hyp_params)
        G_S_list, _ = tsp.QLearning(graphs)
        graphDF_func = self.graphDF(G_S_list)
        df = pd.concat([graphDF_func(i) for i in range(len(G_S_list))], axis=0).reset_index(drop=True)
        output_file = self.newFile(graph_file, "approxS")
        df.to_csv(output_file)
        return df

    def getG_S_rFromCSV(self, file):
        '''
        :param file: file to get (G, S, r) list from
        :return: (G, S, r) list where G is graph, S is calculated permutation, and r is approximation ratio from S
        '''
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
        '''
        :param graph_file: the file to read from or to write to for the walked graphs
        :param graphDist: the graph distribution used to generate graphs (if the file is not already created)
        :param n: the number of graphs to generate (if the file is not already created)
        :return: (G, S, r) list where G is graph, S is calculated permutation, and r is approximation ratio from S
        '''
        try:
            return self.getG_S_rFromCSV(self.newFile(graph_file, "approxS"))
        except FileNotFoundError:
            self.toCSV(graph_file, graphDist, n)
            return self.getG_S_rFromCSV(self.newFile(graph_file, "approxS"))

    def drawGraphs(self, graph_file, graphDist, n):
        '''
        :param graph_file: the file to read from or to write to for the walked graphs
        :param graphDist: the graph distribution used to generate graphs (if the file is not already created)
        :param n: the number of graphs to generate (if the file is not already created)
        :return: None (side effect: saves and shows plots of graphs drawn with optimal path along with calculated path)
        '''
        G_S_r_list = self.getG_S_r(graph_file, graphDist, n)
        fig, axs = plt.subplots(1, n)
        #fig.suptitle("Learned Paths of Graphs over {} Episodes".format(n))

        def graph_plot(i):
            G, S, r = G_S_r_list[i]
            ax = axs[i]
            ordered_points = G.points
            closed = lambda z: z + (z[0],)
            x_true, y_true = tuple(zip(*ordered_points))
            ax.set_title("Episode {}".format(i + 1))
            ax.set_xlabel("Approx Ratio: {}".format(round(r, 2)))
            ax.plot(x_true, y_true, "o")
            x_approx, y_approx = tuple(zip(*[G.points[j] for j in S]))
            ax.plot(closed(x_approx), closed(y_approx), "-", label="Approximated")
            ax.plot(closed(x_true), closed(y_true), "-", label = "Optimal")
            ax.legend()
            #plt.savefig(self.newFile(graph_file, "Graph {} Plot".format(i), ext=".png"))
            #plt.show()
        [graph_plot(i) for i in range(len(G_S_r_list))]
        plt.savefig(self.newFile(graph_file, "Learned Paths", ext=".png"))
        plt.show()



