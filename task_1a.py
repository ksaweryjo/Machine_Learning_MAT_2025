import numpy as np
import pytest
from sklearn.linear_model import Ridge

class RidgeRegr:
    def __init__(self, alpha=0.0, lr=0.01, iters=10000):
        self.alpha = alpha
        # Dodajemy learning rate i maksymalną liczbę iteracji
        self.lr = lr
        self.iters = iters

    def fit(self, X, Y):
        n, m = X.shape
        # Dodajemy bias
        X_b = np.column_stack([np.ones(n), X])
        self.theta = np.zeros(m + 1)

        I = np.eye(m + 1)
        I[0, 0] = 0

        # Minimalizujemy funkcję kosztu (wzór w formie macierzowej)
        # L(theta) = 1/2 * (Y - X_b * theta)^T * (Y - X_b * theta) + 1/2 * alpha * theta^T * I * theta
        for _ in range(self.iters):
            # Grad(L(θ)) = - X_b^T * (Y - X_b * theta) + alpha * I * theta
            gradL = - (X_b.T @ (Y - X_b @ self.theta)) + self.alpha * (I @ self.theta)
            self.theta -= self.lr * gradL # Mnożymy razy lr, aby gradien zbiegał stabilnie

        return self
    
    def predict(self, X):
        k, m = X.shape
        # Dodajemy bias
        X_b = np.column_stack([np.ones(k), X])
        # Wzór jak w z1a
        return X_b @ self.theta


def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)
    
if __name__ == "__main__":
    try:
        test_RidgeRegressionInOneDim()
        print("One Dim Passed")
    except AssertionError:
        print("One Dim Did Not Pass")
        
    try:
        test_RidgeRegressionInThreeDim()
        print("Three Dim Passed")
    except AssertionError:
        print("Three Dim Did Not Pass")