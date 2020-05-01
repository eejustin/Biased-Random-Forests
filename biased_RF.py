import numpy as np
import pandas as pd
from kNN import KNN
from decision_tree import DecisionTree
from random_forest import RandomForest


class BiasedRF:
    def __init__(self, K, p, s):
        self.K = K
        self.p = p
        self.s = s
        self.df = None
        self.isPositive_minor = None


    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)

    def display_imbalance_stats(self):
        # counts for all classes
        counts = self.df['Outcome'].value_counts()

        # get fraudulent and valid cnts
        pos_cnts = counts[1]
        neg_cnts = counts[0]

        self.isPositive_minor = True if pos_cnts<neg_cnts else self.isPositive_minor = False

        # calculate percentage of fraudulent data
        percentage = pos_cnts / (pos_cnts + neg_cnts)
        print(f'Total number of samples: {pos_cnts+neg_cnts} \n'
              f'{pos_cnts} \n' \
              f'{neg_cnts} \n' \
              f'positives takes {percentage*100}%')

    def fill_na(self):
        # replace 0 values at feature columns with mean
        cols = list(self.df.columns.values[:-1])
        self.df[cols] = self.df[cols].replace({0: np.nan})
        self.df = self.df.fillna(self.df.mean())

    def normalization(self):
        self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())

    def seperate_minor(self):
        df_pos = self.df.loc[self.df['Outcome'] == 1]
        df_neg = self.df.loc[self.df['Outcome'] == 0]

        if self.isPositive_minor:
            df_minor = df_pos
            df_major = df_neg
        else:
            df_minor = df_neg
            df_major = df_pos

        return df_minor, df_major

    def create_difficult_area(self):
        df_minor, df_major = self.seperate_minor()
        knn = KNN(df_minor, df_major, self.K)
        difficult_area = knn.get_neighbors()

        return difficult_area

    def construct_RF(self):
        full_dataset = self.df
        difficult_area = self.create_difficult_area()
        critical_dataset = self.df.iloc[difficult_area]

        rf1 = RandomForest(full_dataset, 5, self.s*(1-self.p))
        trees_1 = rf1.construct_trees()
        rf2 = RandomForest(critical_dataset, 5, self.s*self.p)
        trees_2 = rf2.construct_trees()
        trees = trees_1 + trees_2
        return trees






