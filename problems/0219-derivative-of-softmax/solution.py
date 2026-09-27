import  numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	def softmax(x:np.ndarray):
		return np.exp(x) / np.sum(np.exp(x))
	
	n = len(x)
	jac = np.zeros((n, n))
	s = softmax(np.array(x))
	for i in range(n):
		for j in range(n):
			jac[i, j] = s[j] * (1 - s[i]) if i == j else -s[i] * s[j]
	return jac.tolist()