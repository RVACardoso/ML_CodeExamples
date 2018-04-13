import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class adln_onlineSGD(object):
    def __init__(self, eta=0.01, num_iter=10, shuffle=True, random=None):
        self.eta = eta
        self.num_iter = num_iter
        self.shuffle_on = shuffle
        if random:
            np.random.seed(random)

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.cost_ = []
        for _ in range(self.num_iter):
            if self.shuffle:
                X, y = self.shuffle(X, y)
            cost = []
            for x_i, target in zip(X, y):
                cost.append(self.update_weights(x_i, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def shuffle(self, X, y):
        perm = np.random.permutation(len(y))
        return X[perm], y[perm]

    def update_weights(self, x_i, target):
        output = self.weighted_sum(x_i)
        error = target - output
        self.w_ += self.eta * x_i.dot(error)
        cost = error ** 2
        return cost

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
plt.show()

#scaling for optimal performance of algorithm
X_scld = np.copy(X)
X_scld[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_scld[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

#use neuron
adln_on = adln_onlineSGD(num_iter=15, eta=0.01, random=1)
X0 = np.ones((X_scld.shape[0],1))
X_scld = np.hstack((X0,X_scld))
adln_on.fit(X_scld, y)
plt.plot(range(1, len(adln_on.cost_) + 1), adln_on.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

print("Final cost: " + str(adln_on.cost_[-1]))