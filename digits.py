from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from skimage import transform
import sys
from collections import defaultdict

digits = load_digits()
(X_train, X_test, y_train, y_test) = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42)

# Commented below is how to find the right k for our k-neighbors classifier
# We use cross validation with 5 folds
# Best k is 3

# ks = np.arange(2, 10)
# scores = []
# for k in ks:
    # model = KNeighborsClassifier(n_neighbors=k)
    # score = cross_val_score(model, X_train, y_train, cv=5)
    # scores.append(score.mean())

def display_image_and_result(X, y, idx):
    print(f"RESULT SHOULD BE {y[idx]}")

    image = X[idx]
    image = transform.resize(image.reshape(8,8), (32, 32), mode='constant', preserve_range=True).ravel()
    image = image.reshape(32, 32)
    plt.imshow(image, interpolation='nearest')
    plt.show()


class StupidModel:
    def __init__(self, n_neighbors=4):
        self.n_neighbors = n_neighbors

    # "fit" that keeps all data in memory, which is very inefficient but OK for testing purposes
    def fit(self, X, y):
        self.data = X
        self.target = y

    # x is one element
    def predict(self, x):
        distances = []
        for idx, d in enumerate(self.data):
            distances.append((np.linalg.norm(x-d), idx))  # Using Euclidian norm as distance

        distances.sort()

        freq = defaultdict(lambda: 0)
        for _, idx in distances[0:self.n_neighbors]:
            freq[self.target[idx]] += 1

        freq = [(v, k) for k, v in freq.items()]
        freq.sort()

        return freq[0][-1]



model = StupidModel(n_neighbors=3)
model.fit(X_train, y_train)

for idx in range(10):
    test_data = X_test[idx]
    prediction = model.predict(test_data)
    print(f"PREDICTION: {prediction}")
    display_image_and_result(X_test, y_test, idx)




sys.exit(0)

# Below the code that uses the scikit learn model

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


for idx in range(10):
    test_data = X_test[idx]
    print(test_data.reshape(1, -1))
    prediction = model.predict(test_data.reshape(1, -1))

    print(f"PREDICTION: {prediction}")
    display_image_and_result(X_test, y_test, idx)


