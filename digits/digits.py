from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from skimage import transform

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


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


for idx in range(10):
    test_data = X_test[idx]
    print(test_data.reshape(1, -1))
    prediction = model.predict(test_data.reshape(1, -1))

    print(f"PREDICTION: {prediction}")
    display_image_and_result(X_test, y_test, idx)


