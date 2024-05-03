from Analysis_3 import Analysis_3
from EuclideanGraphDistribution import EuclideanGraphDistribution

def f(i):
    if i == 1:
        A = Analysis_3()
        for i, m in zip(range(3), [9, 14, 17]):
            graph_file = "\\".join([A.folder(3), "Graphs_{}".format(i), "Graphs_{}.csv".format(i)])
            graphDist = EuclideanGraphDistribution(pt_number_lim=(m, m))
            A.drawGraphs(graph_file, graphDist, 3)

if __name__ == '__main__':
    f(1)

