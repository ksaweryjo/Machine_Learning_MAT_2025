import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)

def softmax(x):
    f_x = np.exp(x)
    return f_x / np.sum(f_x, axis=0, keepdims=True)

class Layer:
    def __init__(self, n_inputs, n_neurons, learning_rate=0.1, activation="sigmoid"):
        self.W = np.random.randn(n_neurons, n_inputs + 1) * np.sqrt(1.0 / n_inputs)
        self.lr = learning_rate
        self.activ = activation
        self.x = None
        self.a = None
        self.input = None

    def forward(self, x):
        x_bias = np.vstack([np.ones((1, x.shape[1])), x])
        self.input = x_bias
        self.x = self.W @ x_bias
        if self.activ == "sigmoid":
            self.a = sigmoid(self.x)
        elif self.activ == "softmax":
            self.a = softmax(self.x)
        return self.a

    def backward(self, grad_out=None, Y=None):
        if self.activ == 'sigmoid':
            dz = grad_out * sigmoid_prime(self.x)
        elif self.activ == 'softmax':
            dz = self.a - Y

        batch_size = dz.shape[1]
        dW = (dz @ self.input.T) / batch_size
        grad_bias = self.W.T @ dz
        grad_in = grad_bias[1:, :]
        self.W -= self.lr * dW
        return grad_in

class NeuralNet:
    def __init__(self, layers, learning_rate):
        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(Layer(layers[i], layers[i + 1], learning_rate, activation="sigmoid"))
        self.layers.append(Layer(layers[-2], layers[-1], learning_rate, activation="softmax"))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, Y):
        grad = None
        for layer in reversed(self.layers):
            if layer.activ == "softmax":
                grad = layer.backward(Y=Y)
            else:
                grad = layer.backward(grad_out=grad)

    def train_batch(self, X_batch, Y_batch):
        Y_pred = self.forward(X_batch)
        if self.layers[-1].activ == "softmax":
            loss = -np.mean(np.sum(Y_batch * np.log(Y_pred + 1e-8), axis=0))
        else:
            loss = 0.5 * np.mean(np.sum((Y_batch - Y_pred) ** 2, axis=0))

        self.backward(Y_batch)
        return loss

    def fit(self, X, Y, epochs, batch_size, shuffle=True, validation_data=None):
        samples = X.shape[0]
        for ep in range(epochs):
            samples = X.shape[1]
            if shuffle:
                perm = np.random.permutation(samples)
                X = X[:, perm]
                Y = Y[:, perm]
            losses = []
            for i in range(0, samples, batch_size):
                xb = X[:, i:i + batch_size]
                yb = Y[:, i:i + batch_size]
                loss = self.train_batch(xb, yb)
                losses.append(loss)

            avg_loss = np.mean(losses)
            msg = f"Epoch {ep + 1}/{epochs}, loss = {avg_loss:.4f}"
            if validation_data is not None:
                val_X, val_Y = validation_data
                val_pred = self.predict(val_X)
                val_acc = np.mean(val_pred == np.argmax(val_Y, axis=0))
                msg += f", val_acc = {val_acc * 100:.2f}%"
            print(msg)

    def predict(self, X):
        A = self.forward(X)
        return np.argmax(A, axis=0)