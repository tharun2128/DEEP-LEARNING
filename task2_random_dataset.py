# ============================================
# Name    : Tharun
# Roll No : 24BAD126
# Task 2: Train Perceptron on a Random Dataset
# ============================================

import numpy as np

# ---------------- Perceptron Class ----------------
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation(linear_output)

                error = y[i] - prediction

                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, X):
        predictions = []
        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            predictions.append(self.activation(linear_output))
        return np.array(predictions)

# ---------------- Accuracy Function ----------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

# ---------------- Generate Random Dataset ----------------
np.random.seed(42)

# 100 samples, 2 features
X = np.random.randint(0, 10, (100, 2))

# Rule:
# If sum of two features >= 10 → Class = 1
# Else → Class = 0
y = np.array([1 if x[0] + x[1] >= 10 else 0 for x in X])

# ---------------- Train Perceptron ----------------
model = Perceptron(learning_rate=0.1, epochs=100)
model.fit(X, y)

# ---------------- Predict ----------------
predictions = model.predict(X)

# ---------------- Results ----------------
print("=" * 50)
print("Random Dataset")
print("=" * 50)

print("\nFirst 10 Samples:")
print(X[:10])

print("\nActual Labels:")
print(y[:10])

print("\nPredicted Labels:")
print(predictions[:10])

print("\nAccuracy: {:.2f}%".format(accuracy(y, predictions)))

print("\nTraining Completed Successfully!")