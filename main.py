from TSP_RL import TSP_RL
from Analysis_1 import Analysis_1
from TSP_HK import TSP_HK
from WalkedGraphs import WalkedGraphs
import os
import pandas as pd
from Analysis_2 import Analysis_2
from Analysis_3 import Analysis_3
from EuclideanGraphDistribution import EuclideanGraphDistribution

def f(i):
    if i == 1:
        A = Analysis_3()
        for i in range(3):
            graph_file = "\\".join([A.folder(3), "Graphs_{}".format(i), "Graphs_{}.csv".format(i)])
            A.drawGraphs(graph_file, EuclideanGraphDistribution(), 3)

if __name__ == '__main__':
    f(1)

