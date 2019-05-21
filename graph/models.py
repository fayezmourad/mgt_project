from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
import copy
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import sklearn.svm as svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from helpers import *

class Lasso_Regression():
    def __init__(self, alpha, seed=0):
        self.seed=seed
        self.alpha=alpha
        self.clf=Lasso(alpha=alpha, random_state=seed)

    def train(self, input, labels):
        self.clf.fit(input, labels)

    def classify(self, input):
        self.prediction = self.clf.predict(input)
        return self.prediction

    def error(self, input, labels):
        self.prediction = self.clf.predict(input)
        return mean_absolute_error(labels, self.prediction)

    def reset(self):
        self.clf = Lasso(alpha=self.alpha, random_state=self.seed)
        
        
class KNN():
    def __init__(self, n_neighbors=42):
        self.n_neighbors = n_neighbors
        self.clf = KNeighborsRegressor(n_neighbors=n_neighbors)

    def train(self, input, labels):
        self.clf.fit(input, labels)

    def classify(self, input):
        self.prediction = self.clf.predict(input)
        return self.prediction

    def error(self, input, labels):
        self.prediction = self.clf.predict(input)
        return mean_absolute_error(labels, self.prediction)

    def reset(self):
        self.clf = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        
class MLP():
    def __init__(self, seed=0, solver='adam', alpha=1e-8, hidden_layers=(25, 25), lr=1e-4, max_iter=1000):
        self.seed = seed
        self.solver = solver
        self.alpha = alpha
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.max_iter = max_iter

        self.clf = MLPRegressor(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layers,
                                 shuffle=True, max_iter=max_iter, learning_rate_init=lr, random_state=self.seed)

    def train(self, input, labels):
        self.clf.fit(input, labels)

    def classify(self, input):
        self.prediction = self.clf.predict(input)
        return self.prediction

    def error(self, input, labels):
        self.prediction = self.clf.predict(input)
        return mean_absolute_error(labels, self.prediction)

    def reset(self):
        self.clf = MLPRegressor(solver=self.solver, alpha=self.alpha, hidden_layer_sizes=self.hidden_layers,
                                 shuffle=True, max_iter=self.max_iter, learning_rate_init=self.lr, random_state=self.seed)

        
class SVR():
    def __init__(self, kernel, poly_degree=3, seed=0):
        self.kernel = kernel
        self.poly_degree = poly_degree
        self.seed = seed
        if kernel == 'linear':
            self.clf = svm.LinearSVR(random_state=seed) # Can use 'crammer_singer’ but more expensive while not that much better accuracy(only more stable)
        else:
            self.clf = svm.SVR(gamma='auto', kernel=kernel, degree=poly_degree, random_state=seed)

    def train(self, input, labels):
        self.clf.fit(input, labels)

    def classify(self, input):
        self.prediction = self.clf.predict(input)
        return self.prediction

    def error(self, input, labels):
        self.prediction = self.clf.predict(input)
        return mean_absolute_error(labels, self.prediction)

    def reset(self):
        if self.kernel == 'linear':
            self.clf = svm.LinearSVR(random_state=self.seed)
        else:
            self.clf = svm.SVR(gamma='auto', kernel=self.kernel, degree=self.poly_degree,
                               random_state=self.seed)

class Random_Forest():
    def __init__(self, n_estimators, max_depth, criterion='gini', seed=0):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,random_state=seed)

    def train(self, input, labels):
        self.clf.fit(input, labels)

    def classify(self, input):
        self.prediction = self.clf.predict(input)
        return self.prediction

    def error(self, input, labels):
        self.prediction = self.clf.predict(input)
        return mean_absolute_error(labels, self.prediction)

    def reset(self):
        self.clf = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.n_estimators,random_state=self.seed)
