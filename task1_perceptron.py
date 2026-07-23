# ============================================
# Name    : Tharun
# Roll No : 24BAD126
# Task 1: Single Layer Perceptron for AND & OR Logic Gates
# ============================================

import numpy as np

# Single Layer Perceptron Class
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Activation Function (Step Function)
    def activation(self, x):
        return 1 if x >= 0 else 0

    # Training Function
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

    # Prediction Function
    def predict(self, X):
        predictions = []
        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            predictions.append(self.activation(linear_output))
        return np.array(predictions)


# Accuracy Function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


# Input Dataset (Common for AND & OR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# ---------------- AND Logic ----------------
print("=" * 50)
print("AND LOGIC GATE")
print("=" * 50)

y_and = np.array([0, 0, 0, 1])

and_model = Perceptron(learning_rate=0.1, epochs=100)
and_model.fit(X, y_and)

and_predictions = and_model.predict(X)

print("\nInputs:")
print(X)

print("\nExpected Output:")
print(y_and)

print("\nPredicted Output:")
print(and_predictions)

print("\nAccuracy: {:.2f}%".format(accuracy(y_and, and_predictions)))

# ---------------- OR Logic ----------------
print("\n" + "=" * 50)
print("OR LOGIC GATE")
print("=" * 50)

y_or = np.array([0, 1, 1, 1])

or_model = Perceptron(learning_rate=0.1, epochs=100)
or_model.fit(X, y_or)

or_predictions = or_model.predict(X)

print("\nInputs:")
print(X)

print("\nExpected Output:")
print(y_or)

print("\nPredicted Output:")
print(or_predictions)

print("\nAccuracy: {:.2f}%".format(accuracy(y_or, or_predictions)))

print("\nTraining Completed Successfully!")