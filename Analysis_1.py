from TSP_RL import TSP_RL
from functools import reduce
from itertools import product
import time
import pandas as pd
import os
from WalkedGraphs import WalkedGraphs

class Analysis_1:
    def __init__(self, hyp_range_dict, k, n = None, graph_csv = "WalkedGraphs.csv"):
        self.hyp_dict = hyp_range_dict
        self.hyp_names = ["p", "T", "eps", "n", "alpha", "beta", "gamma"]
        hyp_combos = product(*[hyp_range_dict[name] for name in self.hyp_names])
        self.hyp_dicts = [dict(zip(self.hyp_names, hyp_values)) for hyp_values in hyp_combos]
        self.graphs = self.getGraphs(n, graph_csv)
        self.n = len(self.graphs)
        self.train_test_dict = self.get_train_test_dict(k=k)

    def getGraphs(self, n, graph_csv):
        if graph_csv is not None:
            return WalkedGraphs(n, graph_csv).graphs
        else:
            WalkedGraphs(n, graph_csv).getCSV()
            return WalkedGraphs(n, csv="WalkedGraphs.csv").graphs

    def partition(self, k):
        (q, r) = (self.n // k, self.n % k)
        def f(i, j, p):
            return p if i == k else f(i + 1, j + q + int(i < r), p + [list(range(j, j + q + int(i < r)))])
        return f(0, 0, [])

    def get_train_test_dict(self, k):
        partition = self.partition(k)
        train_index = lambda i: reduce(lambda l1, l2: l1 + l2, partition[:i] + partition[(i+1):])
        test_index = lambda i: partition[i]
        return dict([(i, (train_index(i), test_index(i))) for i in range(k)])

    def approx_ratio(self, model, Theta):
        def f(G):
            S = model.calculateWalk(Theta, G)
            return G.tourDistance(S) / G.distance
        return f

    def hyp_error(self, hyp_dict):
        hyp_params = [hyp_dict[k] for k in self.hyp_names]
        tsp = TSP_RL(*hyp_params)
        def fold_error(fold):
            train, test = self.train_test_dict[fold]
            trainGs, testGs = [self.graphs[i] for i in train], [self.graphs[i] for i in test]
            _, Theta = tsp.QLearning(trainGs)
            approx_ratio_func = self.approx_ratio(tsp, Theta)
            return 1 / len(test) * sum([approx_ratio_func(G) for G in testGs])

        return 1 / len(self.train_test_dict) * sum([fold_error(fold) for fold in self.train_test_dict.keys()])

    def getErrorDf(self, append = True):
        def error_row(i):
            hyp_dict = self.hyp_dicts[i]
            hyps = [hyp_dict[name] for name in self.hyp_names]
            error = self.hyp_error(hyp_dict)
            row = hyps + [error]
            df = pd.DataFrame.from_dict(data = {i: row}, orient = "index", columns = self.hyp_names + ["Error"])
            df.to_csv("\\".join([os.getcwd(), "Error", "Error_{}.csv".format(i)]))
            return pd.DataFrame.from_dict(data = {i: row}, orient = "index", columns = self.hyp_names + ["Error"])

        start_time = time.time()

        if append:
            prev_df = pd.read_csv("\\".join([os.getcwd(), "Error", "Error.csv"]), index_col=0)
            i = prev_df.index[-1] + 1
            new_row = error_row(i)
            df = pd.concat([prev_df, new_row], axis=0)
            df.to_csv("\\".join([os.getcwd(), "Error", "Error.csv"]))
        else:
            df = error_row(0)
            df.to_csv("\\".join([os.getcwd(), "Error", "Error.csv"]))

        print("Time Elapsed: {} Minutes".format((time.time() - start_time) / 60))
        return df










