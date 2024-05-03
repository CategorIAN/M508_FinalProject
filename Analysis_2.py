from Analysis_1 import Analysis_1
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import os

class Analysis_2(Analysis_1):
    def __init__(self, graph_file = None, graphDist = None, n = None):
        '''
        :param file: the file to read from or to write to for the walked graphs
        :param graphDist: the graph distribution used to generate graphs (if the file is not already created)
        :param n: the number of graphs to generate (if the file is not already created)
        '''
        hyp_range_dict = {"p": [3, 4], "T": [2, 1], "eps": [0.01, 0.05], "n": [2, 3],
                          "alpha": [0.01, 0.1], "beta": [5, 10], "gamma": [0.9, 1]}
        graph_file = "\\".join([os.getcwd(), "Analysis_1", "OurWalkedGraphs.csv"]) if graph_file is None else graph_file
        super().__init__(hyp_range_dict, 4, graph_file = graph_file, graphDist=graphDist, n = n)
        self.error_file = "\\".join([self.folder(1), "Error.csv"])

    def groupedErrorDf(self):
        '''
        :return: function that takes a hyperparameter and returns dataframe showing mean error grouped by hyperparameter
        '''
        df = pd.read_csv(self.error_file, index_col=0)
        return lambda name: df.groupby(by = [name])["Error"].mean()

    def groupedErrorDfs(self):
        '''
        :return: dictionary that has key-value pairs (hyp, groupedErrorDf(hyp))
        '''
        groupedErrorDf_func = self.groupedErrorDf()
        def appendDf(df_dict, name):
            grouped_df = groupedErrorDf_func(name)
            grouped_df.to_csv("\\".join([self.folder(2), "Error_By_{}.csv".format(name)]))
            return df_dict | {name: grouped_df}
        return reduce(appendDf, self.hyp_names, {})

    def plots(self):
        '''
        :return: None (side effect: shows bar charts of average approximation error grouped by each hyperparameter)
        '''
        df_dict = self.groupedErrorDfs()
        def plot(hyp_index):
            fig, axs = plt.subplots(2, 2, figsize = (4, 2))
            hyp_names_title = [self.hyp_names[i] for i in hyp_index]
            fig.suptitle("Average Approximation Ratio By Hyperparameter (For {})".format(hyp_names_title))
            for i in range(len(hyp_index)):
                r, c = i // 2, i % 2
                hyp = self.hyp_names[hyp_index[i]]
                df = df_dict[hyp]
                ax = axs[r, c]
                ax.set_xlabel(hyp)
                ax.set_ylim([1, 1.1])
                ax.bar([str(x) for x in df.index], df)
            plt.show()
        for index in [list(range(4)), list(range(4, 7))]:
            plot(index)

    def bestParams(self):
        '''
        :return: dictionary that shows the best assignment of values for each hyperparameter
        '''
        df_dict = self.groupedErrorDfs()
        def best_param(name):
            grouped_df = df_dict[name]
            min_error = min(grouped_df)
            return grouped_df.loc[lambda df: df == min_error].index[0]
        return dict([(name, best_param(name)) for name in df_dict.keys()])

    def bestRows(self):
        '''
        :return: error dataframe filtered by the rows having a perfect approximation ratio
        '''
        df = pd.read_csv(self.error_file, index_col=0)
        best_rows = df.loc[lambda df: df["Error"] == 1]
        best_rows.to_csv("\\".join([self.folder(2), "Error_Best.csv"]))
        return best_rows

