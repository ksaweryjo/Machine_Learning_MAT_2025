import mnist_loader
import numpy as np
from Neural_Net_277513 import NeuralNet

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

X_train = np.hstack([x for x, y in training_data]) # (784, 50000)
X_test = np.hstack([x for x, y in test_data]) # (784, 10000)
Y_train = np.hstack([y for x, y in training_data]) # (10, 50000)
Y_test = np.array([y for x, y in test_data]) # (10000,)
val_X = np.hstack([x for x, y in validation_data]) # (784, 10000)
val_Y = np.array([y for x, y in validation_data]) # (10000,)
val_Y_fix = np.zeros((10, val_Y.shape[0])) # (10, 10000)
val_Y_fix[val_Y, np.arange(val_Y.shape[0])] = 1

layers = [784, 128, 256, 10]
nnet = NeuralNet(layers, learning_rate=0.01)
nnet.fit(X_train, Y_train, epochs=30, batch_size=32, validation_data=(val_X, val_Y_fix))
pred = nnet.predict(X_test)
acc = np.mean(pred == Y_test)
print(f"Accuracy: {acc * 100:.2f} %")
