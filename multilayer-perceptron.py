import tensorflow as t
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = t.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


print(x_train[0])

# For normalization image pixel values are divided by 255
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

# To understand the structure of dataset
print("Train feature matrix:", x_train.shape)
print("Train target matrix:", y_train.shape)
print("Test target matrix:", x_test.shape)
print("Test target matrix:", y_test.shape)

# Uncomment to see what the input looks like
"""
fig, ax = plt.subplots(10, 10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28, 28),
                        aspect='auto')
        k += 1
plt.show()
"""

model = Sequential([
    # Reshape 28 row * 28 column data to 28*28 rows
    Flatten(input_shape=(28, 28)),

    # Dense layer 1
    Dense(256, activation='sigmoid'),

    # Dense layer 2
    Dense(128, activation='sigmoid'),

    # Output layer
    Dense(10, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=2000, validation_split=0.2)

def display_image(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()

# This predicts a single occurence
# one_prediction = model.predict(np.array([x_test[0]]))

predictions = model.predict(x_test)

goods = 0
SHOW_UP_TO = -1

for idx, prediction in enumerate(predictions):
    predicted = np.argmax(prediction)
    expected = y_test[idx]

    if predicted == expected:
        goods += 1

    if idx <= SHOW_UP_TO:
        print(f"Expected {expected}, predicted {predicted}")
        display_image(x_test[idx])

print(f"Overall accuracy {goods / len(predictions)}")





