import pandas as pd
import numpy as np

class KNN:
    def __init__(self, df_minor, df_major, num_neighbors):
        self.df_minor = df_minor
        self.df_major = df_major
        self.num_neighbors = num_neighbors

    def _euclidean_distance(self, en1, en2):
        dist = np.linalg.norm(en1 - en2)
        return dist

    # Locate the most similar neighbors
    def get_neighbors(self):

        known_major_index = []
        known_minor_index = []
        for i, minor in self.df_minor.iterrows():
            known_minor_index.append(i)
            distances = {}
            for j, major in self.df_major.iterrows():
                dist = self._euclidean_distance(minor, major)
                distances[j] = dist
            sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])
            # print(f'for entry{i}: distances are: {sorted_distances}')
            nn = sorted_distances[:self.num_neighbors]

            for index, dist in nn:
                if index not in known_major_index:
                    known_major_index.append(index)

        return known_major_index + known_minor_index





