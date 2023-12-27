import numpy as np


data = [
    (np.array([1.66, 1.56]), 1),
    (np.array([2, 1.5]), 0)
]


weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])
alpha = 0.3

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

def error(prediction, target):
    return (prediction - target) ** 2

def gradient(prediction, target):
    return 2 * (prediction - target)


print(">>>>> NO TRAINING")
for input_vector, target in data:
    prediction = make_prediction(input_vector, weights_1, bias)
    err = error(prediction, target)
    print(f"The prediction result is: {prediction} with error {err}")

for input_vector, target in data:
    prediction = make_prediction(input_vector, weights_1, bias)
    err = error(prediction, target)
    weights_1 -= alpha * gradient(prediction, target)

print(">>>>> AFTER TRAINING")
for input_vector, target in data:
    prediction = make_prediction(input_vector, weights_1, bias)
    err = error(prediction, target)
    print(f"The prediction result is: {prediction} with error {err}")



