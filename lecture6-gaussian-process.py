# reuse rbf_kernel and gram_matrix from the last lecture
sigma2 = 0.1                         # noise variance σ²

K = gram_matrix(X, X)                # K_ff
K_s = gram_matrix(X_star, X)         # K_f*f
K_ss = gram_matrix(X_star, X_star)   # K_f*f*
A = K + sigma2 * np.eye(len(X))      # K_ff + σ²I

mean = K_s @ np.linalg.solve(A, y)
cov = K_ss - K_s @ np.linalg.solve(A, K_s.T)
std = np.sqrt(np.diag(cov))
