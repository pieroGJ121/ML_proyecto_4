import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from collections import Counter


class Nodo:
    def __init__(self, threshold=None, value=None, label=None):
        self.label = label
        self.threshold = threshold
        self.children = {}
        self.value = value
        self.feature = None

    def is_leaf(self):
        return self.value is not None

    def isTerminal(self):
        T = len(np.unique(self.Y))
        return T == 1

    def Entropy(self, Y):
        counts = np.unique(Y, return_counts=True)[1]
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def Gini(self):
        counts = np.unique(self.Y, return_counts=True)[1]
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities**2)


class DT:
    def __init__(self, max_depth=100):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.root = self.insert_node(X, Y)

    def insert_node(self, X, Y, depth=0):
        num_labels = len(np.unique(Y))
        n_feats = X.shape[1]
        # print("n_feats")
        # print(n_feats)
        # parada
        if depth >= self.max_depth or num_labels == 1:
            leaf_value = self._most_common_label(Y)
            return Nodo(value=leaf_value)

        feats_randoms = np.random.choice(n_feats, n_feats, replace=False)

        # buscar la mejor caracteristica
        best_feature, best_threshold = self._best_split(X, Y, feats_randoms)
        # print("best_feature")
        # print(best_feature)
        # print("best_threshold")
        # print(best_threshold)
        node = Nodo(threshold=best_threshold)
        node.feature = best_feature
        node.threshold = best_threshold
        # dividir los datos
        X_column = X[:, best_feature]
        left_idxs = np.argwhere(X_column <= best_threshold).flatten()
        right_idxs = np.argwhere(X_column > best_threshold).flatten()

        node.children["left"] = self.insert_node(X[left_idxs], Y[left_idxs], depth + 1)
        node.children["right"] = self.insert_node(
            X[right_idxs], Y[right_idxs], depth + 1
        )
        return node
        # left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        # right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        # return Node(best_feature, best_thresh, left, right)

    # buscamos el label que mas se repite
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _best_split(self, X, y, feats_randoms):
        best_gini = 1
        best_feature = None
        best_threshold = None

        for feat_idx in feats_randoms:
            X_column = X[:, feat_idx]
            threshold, gini = self.Los_Gimmis(X_column, y)
            # print("threshold")
            # print(threshold)
            # print("gini")
            # print(gini)
            if gini < best_gini:
                best_gini = gini
                best_feature = feat_idx
                best_threshold = threshold

        return best_feature, best_threshold

    def Los_Gimmis(self, X_column, y):
        # print("X_column")
        # pasar datos a un dataframe
        X_column_sort = pd.Series(X_column)
        X_column = pd.Series(X_column)
        # print("X_column")
        # ordenar los valores
        X_column_sort = X_column_sort.sort_values()
        # print(X_column_sort)
        adyacent_average = (X_column_sort[:-1].values + X_column_sort[1:].values) / 2
        # print("adyacent_average")
        # print(adyacent_average)
        gimis = {}
        for punto_medio in adyacent_average:
            X_izq = X_column[X_column <= punto_medio]
            X_der = X_column[X_column > punto_medio]
            # print("X_izq")
            # print(X_izq)
            # print("X_der")
            # print(X_der)
            gini_izq = self.Gini(y[X_izq.index])
            gini_der = self.Gini(y[X_der.index])
            # print("gini_izq")
            # print(gini_izq)
            # print("gini_der")
            # print(gini_der)
            ponderado = (len(X_izq) / len(X_column)) * gini_izq + (
                len(X_der) / len(X_column)
            ) * gini_der
            # print("ponderado")
            # print(ponderado)
            gimis[punto_medio] = ponderado
        threshold = min(gimis, key=gimis.get)  # mejor punto medio o threshold
        return threshold, gimis[threshold]

    def Gini(self, Y):
        counts = np.unique(Y, return_counts=True)[1]
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities**2)

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().transpose()
        return self.predict_recursivo(X, self.root)

    def predict_recursivo(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_recursivo(x, node.children["left"])
        return self.predict_recursivo(x, node.children["right"])

    def predict_2(self, X):
        return np.array([self.predict_recursivo(x, self.root) for x in X])
