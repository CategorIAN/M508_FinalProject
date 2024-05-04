from Analysis_3 import Analysis_3
from EuclideanGraphDistribution import EuclideanGraphDistribution
import pandas as pd
import os

def f(i):
    if i == 1:
        A = Analysis_3()
        for i, m in zip(range(3), [9, 14, 17]):
            graph_file = "\\".join([A.folder(3), "Graphs_{}".format(i), "Graphs_{}.csv".format(i)])
            graphDist = EuclideanGraphDistribution(pt_number_lim=(m, m))
            A.drawGraphs(graph_file, graphDist, 3)
    if i == 2:
        graph_file = "\\".join([os.getcwd(), "Analysis_1", "OurWalkedGraphs.csv"])
        graph_df = pd.read_csv(graph_file, index_col=0)
        print(graph_df.loc[36:, :].to_latex())
    if i == 3:
        graph_file = "\\".join([os.getcwd(), "Analysis_2", "Error_Best.csv"])
        graph_df = pd.read_csv(graph_file, index_col=0)
        print(graph_df.to_latex())

if __name__ == '__main__':
    f(3)

