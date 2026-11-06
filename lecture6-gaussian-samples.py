import numpy as np

x = np.random.randn(100, 2)    # sample 100 observations (D = 2)

m = [-2.0, 1.0]                # mean vector of y
V = [[1.0, 0.7], [0.7, 2.0]]   # covariance matrix of y

L = np.linalg.cholesky(V)      # matrix square root (Cholesky factor)
y = (x @ L.T) + m              # transform x samples into y
