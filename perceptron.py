import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, n_features, learning_rate=0.01, n_iterations=1000):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def activation_function(self, z):
        return 1 if z > 0 else 0

    def predict(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def fit(self, X, y):
        for _ in range(self.n_iterations):
            for index, x_i in enumerate(X):
                y_pred = self.predict(x_i)
                update = self.learning_rate * (y[index] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def score(self, X, y):
        predictions = [self.predict(x) for x in X]
        return np.mean(predictions == y)

iris = load_iris()
X = iris.data[:, :2]  
y = iris.target

X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron(n_features=X_train.shape[1], learning_rate=0.01, n_iterations=1000)
perceptron.fit(X_train, y_train)

print("Test Accuracy:", perceptron.score(X_test, y_test))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = np.array([perceptron.predict(np.array([x1, x2])) for x1, x2 in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)  
plt.contour(xx, yy, Z, colors='k', linewidths=1)  

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', edgecolor='k', label='Train Data', cmap=plt.cm.RdYlBu)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', edgecolor='k', label='Test Data', cmap=plt.cm.RdYlBu)

plt.title('Perceptron Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(False)
plt.show()
