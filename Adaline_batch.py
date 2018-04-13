import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
    def __init__(self, eta=0.01, num_iter=50):
        self.num_iter = num_iter
        self.eta = eta

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.cost_ = []
        for i in range(self.num_iter):
            output = self.weighted_sum(X)
            errors = (y - output)
            self.w_ += self.eta * X.T.dot(errors)
            cost = (errors ** 2).sum()
            self.cost_.append(cost)
        return self

    def weighted_sum(self, X):
        return np.dot(X, self.w_) 

    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

#get data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = data.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = data.iloc[0:100, [0, 2]].values
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='red', marker='o', label='setosa')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#scaling for optimal performance of algorithm
X_scld = np.copy(X)
X_scld[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_scld[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

#use neuron
adln = AdalineGD(num_iter=20, eta=0.005)
X0 = np.ones((X_scld.shape[0],1))
X_scld = np.hstack((X0,X_scld))
adln.fit(X_scld, y)
plt.plot(range(1, len(adln.cost_) + 1), adln.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

print("Final cost: " + str(adln.cost_[-1]))