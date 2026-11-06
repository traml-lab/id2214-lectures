import numpy as np

# prepare some synthetic data
X = np.array([ [1.0, 25.0], [2.5, 35.2], [5.0, 50.0], [4.5, 45.0] ])
y = np.array([0.2, 0.35, 1.0, 0.6])

# normalize (keep the statistics for prediction)
X_mean, X_std, y_mean = X.mean(axis=0), X.std(axis=0), y.mean()
y = y - y_mean
X = (X - X_mean) / X_std

def rbf_kernel(a, b, lengthscale=1.0):
    return np.exp(-0.5 * np.sum((a - b)**2) / lengthscale**2)

def gram_matrix(A, B):
    return np.array([[rbf_kernel(a, b) for b in B] for a in A])

K = gram_matrix(X, X)  # N x N Gram matrix
beta = np.linalg.solve(K, y)

X_star = (np.array([[3.0, 40.0]]) - X_mean) / X_std  # new input, same scaling
k_star = gram_matrix(X_star, X)                      # M x N

f_star = k_star @ beta + y_mean
print(f_star)
