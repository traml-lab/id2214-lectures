import numpy as np

# prepare some synthetic data
X = np.array([ [1.0, 25.0], [2.5, 35.2], [5.0, 50.0], [4.5, 45.0] ])
y = np.array([0.2, 0.35, 1.0, 0.6])

# rescale to zero mean
y = y - y.mean()
X = X - X.mean(axis=0)

# compute optimal w
Xt = X.transpose()  # precompute transposed X
A = Xt @ X          # multiply Xt and X
b = Xt @ y          # multiply Xt and y

w_opt = np.linalg.solve(A, b)  # solve linear system A*w = b for w

# perform predictions
Xstar = np.array([ [3.0, 40.0] ])
yhat = (Xstar - X_mean) @ w_opt + y_mean  # same centring as training data

print(yhat)
