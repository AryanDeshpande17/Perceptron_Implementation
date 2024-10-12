import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_features, learning_rate=0.01, n_iterations=1000):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def activation_function(self, z):
        return np.where(z > 0, 1, 0)

    def predict(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def fit(self, X, y):
        for i in range(self.n_iterations):
            for index, x_i in enumerate(X):
                y_pred = self.predict(x_i)
                update = self.learning_rate * (y[index] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def score(self, X, y):
        predictions = [self.predict(x) for x in X]
        return np.mean(predictions == y)

class OneVsAllPerceptron:
    def __init__(self, n_classes, n_features, learning_rate=0.01, n_iterations=1000):
        self.n_classes = n_classes
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.classifiers = []

    def fit(self, X, y):
        for c in range(self.n_classes):
            binary_y = (y == c).astype(int)
            perceptron = Perceptron(self.n_features, self.learning_rate, self.n_iterations)
            perceptron.fit(X, binary_y)
            self.classifiers.append(perceptron) 

    def predict(self, X):
        class_predictions = np.array([classifier.predict(X) for classifier in self.classifiers])
        return np.argmax(class_predictions, axis=0)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ovr_perceptron = OneVsAllPerceptron(n_classes=3, n_features=X.shape[1])
ovr_perceptron.fit(X_train, y_train)

predictions = ovr_perceptron.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f'One-vs-All Perceptron Accuracy: {accuracy * 100:.2f}%')

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.title('PCA of Iris Dataset with One-vs-All Perceptron')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Class Labels')
plt.show()



