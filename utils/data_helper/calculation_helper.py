from sklearn.linear_model import LinearRegression
import numpy as np

def calculate_slope(y):
    X = np.arange(1, len(y)+1).reshape(-1, 1)
    # y = 1 * x_0 + 2 * x_1 + 3
    reg = LinearRegression().fit(X, y)
    return reg.coef_[0]