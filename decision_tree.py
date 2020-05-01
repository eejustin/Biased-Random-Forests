import pandas as pd
import numpy as np


class TreeNode:
    def __init__(self, attr=None, val=None, gini=None):
        self.attr = attr
        self.val = val
        self.gini = gini
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, df):
        self.df = df
        self.root = None

    # Split a dataset based on an attribute and an attribute value
    def split(self, df, attr, val):
        df_left = df.loc[df[attr] < val].drop([attr], axis=1)
        df_right = df.loc[df[attr] >= val].drop([attr], axis=1)

        return df_left, df_right

    # For a particular column(attribute), from all the unique values, find all threshold values to split data
    def find_split_val(self, vals):
        ans = []
        vals.sort()
        for i in range(1, len(vals)):
            ans.append((vals[i] - vals[i - 1]) / 2 + vals[i - 1])
        return ans

    # Obtain count of positive and negative samples
    def get_pos_vs_neg(self, df):
        pos_vs_neg = {"pos": 0, "neg": 0}

        for index, row in df.iterrows():
            if row['Outcome'] == 1:
                pos_vs_neg['pos'] += 1
            else:
                pos_vs_neg['neg'] += 1

        return pos_vs_neg

    # Calculate Gini index for before or after a split
    def get_gini(self, stats, sample_size):

        weighted_gini = 0
        for side in stats:
            n_pos = side['pos']
            n_neg = side['neg']
            count = n_pos + n_neg
            gini = 1 - (n_pos / count) ** 2 - (n_neg / count) ** 2
            weighted_gini += gini * count / sample_size

        return weighted_gini

    # Try to find the best attribute and value to split the node
    def find_best_split(self, df):
        attribute_ginis = {}
        for attr in df.columns.values[:-1]:
            gini_split = {}
            vals = df[attr].unique()
            if len(vals) == 1: continue
            splits = self.find_split_val(vals)

            for val in splits:
                df_left, df_right = self.split(df, attr, val)
                pos_vs_neg_left = self.get_pos_vs_neg(df_left)
                pos_vs_neg_right = self.get_pos_vs_neg(df_right)
                gini_split[val] = self.get_gini([pos_vs_neg_left, pos_vs_neg_right], df.shape[0])

            sorted_gini_split = sorted(gini_split.items(), key=lambda kv: kv[1])
            best_val_gini_pair = sorted_gini_split[0]
            attribute_ginis[attr] = best_val_gini_pair

        sorted_attribute_ginis = sorted(attribute_ginis.items(), key=lambda kv: kv[1][1])

        return sorted_attribute_ginis[0]

    # split the tree to the left and right, if and only if splitting make sense
    def grow_tree(self, node, df):
        if len(df.columns.values) == 1 or node.gini == 0.0:
            return

        best = self.find_best_split(df)
        best_attr = best[0]
        best_val = best[1][0]
        best_gini = best[1][1]

        if best_gini >= node.gini:
            return

        df_left, df_right = self.split(df, best_attr, best_val)
        node.attr = best_attr
        node.val = best_val

        stats_left = self.get_pos_vs_neg(df_left)
        gini_left = self.get_gini([stats_left], df_left.shape[0])
        node.left = TreeNode(gini=gini_left)
        self.grow_tree(node.left, df_left)

        stats_right = self.get_pos_vs_neg(df_right)
        gini_right = self.get_gini([stats_right], df_right.shape[0])
        node.right = TreeNode(gini=gini_right)
        self.grow_tree(node.right, df_right)

    # Construct a decision tree with the root node of root
    def construct_tree(self):

        best = self.find_best_split(self.df)
        stats = self.get_pos_vs_neg(self.df)
        gini = self.get_gini([stats], self.df.shape[0])
        root = TreeNode(gini=gini)
        self.grow_tree(root, self.df)
        self.root = root


