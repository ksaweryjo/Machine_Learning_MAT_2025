import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

class LinearRegr:
    def fit(self, X, Y):
        n, m = X.shape
        # X_b będzie macierzą X z dodaną kolumną jednynek, czyli biasem
        X_b = np.column_stack([np.ones(n), X])
        # theta = (X^T * X)^(-1) * X^T * Y
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y
        return self
    
    def predict(self, X):
        k, m = X.shape
        # Dodajemy bias
        X_b = np.column_stack([np.ones(k), X])
        # y_pred_i = theta_0·1 + theta_1·x_i1 + theta_2·x_i2 + ... + theta_m·x_im dla i=1..k
        return X_b @ self.theta

def test_RegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,2,10]).reshape((3,1))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    assert list(actual) == pytest.approx(list(expected))

def test_RegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    assert list(actual) == pytest.approx(list(expected))
    
if __name__ == "__main__":
    try:
        test_RegressionInOneDim()
        print("One Dim Passed")
    except AssertionError:
        print("One Dim Did Not Pass")

    try:
        test_RegressionInThreeDim()
        print("Three Dim Passed")
    except AssertionError:
        print("Three Dim Did Not Pass")