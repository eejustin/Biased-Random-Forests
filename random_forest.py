import pandas as pd
import numpy as np
from decision_tree import DecisionTree, TreeNode
import random

class RandomForest:
    def __init__(self, df, num_attributes, num_trees):
        self.df = df
        self.num_attributes = num_attributes
        self.num_trees = num_trees

    def boot_strapping(self):
        idx = self.df.index
        selection = random.choices(idx, k=len(idx))
        df_bs = self.df.iloc[selection]
        return df_bs

    def bagging(self, df):
        cols = df.columns.values
        selection = random.sample(cols, k=self.num_attributes)
        df_bg = df[selection]
        return df_bg

    def construct_trees(self):
        trees = []
        df_bs = self.boot_strapping()
        for i in range(self.num_trees):
            df_bg = self.bagging(df_bs)
            trees.append(DecisionTree(df_bg))
        return trees









