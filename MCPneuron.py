import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    #set learning rate and number of iterations
    def __init__(self, eta=0.1, num_iter=10):
        self.num_iter = num_iter
        self.eta = eta

    def fit(self, X, y):
        self.w_ = np.zeros((X.shape[1],1))
        self.errors_ = []
        for _ in range(self.num_iter):
            errors = 0
            for x_i, y_i in zip(X, y):
                update = self.eta * (y_i - self.get_class(x_i))
                self.w_ += np.multiply(update, x_i.reshape((3,1)))
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def get_class(self, x):
        weighted_sum = np.dot(x, self.w_)
        return np.where(weighted_sum >= 0.0, 1, -1)

#get data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = data.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = data.iloc[0:100, [0, 2]].values
plt.scatter(X[y==-1,0], X[y==-1, 1], color='red', marker='o', label='setosa')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.show()
X0 = np.ones((100,1))
X = np.hstack((X0,X))

#use neuron
pcpn = Perceptron(eta=0.2, num_iter=7)
pcpn.fit(X, y)
plt.plot(range(1, len(pcpn.errors_) + 1), pcpn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

print("Final number of missclassifications " + str(pcpn.errors_[-1]))
