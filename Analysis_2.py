from Analysis_1 import Analysis_1
import pandas as pd
from functools import reduce
from TSP_RL import TSP_RL

class Analysis_2(Analysis_1):
    def __init__(self):
        hyp_range_dict = {"p": [3, 4], "T": [2, 1], "eps": [0.01, 0.05], "n": [2, 3],
                          "alpha": [0.01, 0.1], "beta": [5, 10], "gamma": [0.9, 1]}
        super().__init__(hyp_range_dict, 4)


    def groupedErrorDf(self, file):
        df = pd.read_csv(file, index_col=0)
        return lambda name: df.groupby(by = [name])["Error"].mean()

    def groupedErrorDfCSVs(self, folder, input_folder_file):
        groupedErrorDf_func = self.groupedErrorDf("\\".join([folder, input_folder_file]))
        def appendDf(df_dict, name):
            grouped_df = groupedErrorDf_func(name)
            grouped_df.to_csv("\\".join([folder, "Error_By_{}.csv".format(name)]))
            return df_dict | {name: grouped_df}

        return reduce(appendDf, self.hyp_names, {})

    def bestParams(self, folder, input_folder_file):
        df = pd.read_csv("\\".join([folder, input_folder_file]), index_col=0)
        best_rows = df.loc[lambda df: df["Error"] == 1]
        best_rows.to_csv("\\".join([folder, "Error_Best.csv"]))
        return best_rows

