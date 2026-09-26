import numpy as np

def calculate_correlation_matrix(X, Y=None):
	X = np.asarray(X, dtype=np.float64)
	Y = X if Y is None else np.asarray(Y, dtype=np.float64)

	X_centered = X - X.mean(axis=0, keepdims=True)
	Y_centered = Y - Y.mean(axis=0, keepdims=True)

	# compute standard deviation
	stdX = np.sqrt(np.sum(X_centered ** 2, axis=0))
	stdY = np.sqrt(np.sum(Y_centered ** 2, axis=0))

	std_mat = np.outer(stdX, stdY)

	# compute covariance
	cov_mat = X_centered.T @ Y_centered

	cor = cov_mat / std_mat

	return  cor