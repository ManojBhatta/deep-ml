import  numpy as np
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	z = np.array(logits)
	# z -= np.max(logits)

	s = np.exp(z) / np.sum(np.exp(z))

	y = np.zeros_like(z)
	y[target] = 1

	grads = (s - y)
	return  grads.tolist()



