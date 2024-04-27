from TSP_RL import TSP_RL
from RegularPolygon import RandomRegularPolygon, RegularPolygon
from Analysis_1 import Analysis_1
from TSP_HK import TSP_HK
from RandomEuclideanGraph import RandomEuclideanGraph
from WalkedGraphs import WalkedGraphs
import os
import pandas as pd
from Analysis_2 import Analysis_2
from Analysis_3 import Analysis_3

def f(i):
    if i == 1:
        A = Analysis_2()
        print(A.plots())
        print(A.bestRows())
    if i == 2:
        A = Analysis_3()
        A.getG_S_CSV()
    if i == 3:
        A = Analysis_3()
        A.drawGraphs()

if __name__ == '__main__':
    f(3)

