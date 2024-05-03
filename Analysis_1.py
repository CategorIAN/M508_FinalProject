from TSP_RL import TSP_RL
from functools import reduce
from itertools import product
import time
import pandas as pd
import os
from WalkedGraphs import WalkedGraphs

class Analysis_1:
    def __init__(self, hyp_range_dict, k, graph_file = None, graphDist = None, n = None):
        '''
        :param hyp_range_dict: the range of values to use for each hyperparameter
        :param k: the number of parts to use for our partition
        :param graph_file: the file to use for our graph data
        :param graphDist: the graph distribution used to generate graphs (if the file is not already created)
        :param n: the number of graphs to generate (if the file is not already created)
        '''
        self.folder = lambda i: "\\".join([os.getcwd(), "Analysis_{}".format(i)])
        graph_file = "\\".join([self.folder(1), "WalkedGraphs.csv"]) if graph_file is None else graph_file
        self.hyp_range_dict = hyp_range_dict
        self.hyp_names = ["p", "T", "eps", "n", "alpha", "beta", "gamma"]
        hyp_combos = product(*[hyp_range_dict[name] for name in self.hyp_names])
        self.hyp_dicts = [dict(zip(self.hyp_names, hyp_values)) for hyp_values in hyp_combos]
        self.graphs = WalkedGraphs(graph_file, graphDist, n).graphs
        self.n = len(self.graphs)
        self.train_test_dict = self.get_train_test_dict(k=k)

    def partition(self, k):
        '''
        :param k: the number of parts to use for our partition
        :return: a partition of our graph indices
        '''
        (q, r) = (self.n // k, self.n % k)
        def f(i, j, p):
            return p if i == k else f(i + 1, j + q + int(i < r), p + [list(range(j, j + q + int(i < r)))])
        return f(0, 0, [])

    def get_train_test_dict(self, k):
        '''
        :param k: the number of parts to use for our partition
        :return: a dictionary that has key value pairs of (i, train_index(i), test_index(i)) for each i in our k parts
        '''
        partition = self.partition(k)
        train_index = lambda i: reduce(lambda l1, l2: l1 + l2, partition[:i] + partition[(i+1):])
        test_index = lambda i: partition[i]
        return dict([(i, (train_index(i), test_index(i))) for i in range(k)])

    def approx_ratio(self, model, Theta):
        '''
        :param model: TSP Q Learning model to use
        :param Theta: object consisting of list of theta weights to use for Q function to approximate best walk
        :return: approximation ratio for calculated walk
        '''
        def f(G):
            S = model.calculateWalk(Theta, G)
            return G.tourDistance(S) / G.distance
        return f

    def hyp_error(self, hyp_dict):
        '''
        :param hyp_dict: assignment of hyperparameters to use for the model
        :return: average error, as approximation ratio, of each fold
        '''
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
        '''
        :param append: the decision of whether to append to a preexisting error dataframe
        :return: an error dataframe that has been made from a preexisting error dataframe or made from scratch
        '''
        def error_row(i):
            hyp_dict = self.hyp_dicts[i]
            hyps = [hyp_dict[name] for name in self.hyp_names]
            error = self.hyp_error(hyp_dict)
            row = hyps + [error]
            df = pd.DataFrame.from_dict(data = {i: row}, orient = "index", columns = self.hyp_names + ["Error"])
            df.to_csv("\\".join([self.folder(1), "Error_{}.csv".format(i)]))
            return pd.DataFrame.from_dict(data = {i: row}, orient = "index", columns = self.hyp_names + ["Error"])

        start_time = time.time()

        if append:
            prev_df = pd.read_csv("\\".join([os.getcwd(), "Error", "Error.csv"]), index_col=0)
            i = prev_df.index[-1] + 1
            new_row = error_row(i)
            df = pd.concat([prev_df, new_row], axis=0)
            df.to_csv("\\".join([self.folder(1), "Error.csv"]))
        else:
            df = error_row(0)
            df.to_csv("\\".join([self.folder(1), "Error.csv"]))

        print("Time Elapsed: {} Minutes".format((time.time() - start_time) / 60))
        return df










