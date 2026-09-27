import  numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	sigmoid = lambda x: 1 / (1 + np.exp(-x))
	d_sigmoid = lambda x: sigmoid(x) * ( 1 - sigmoid(x))

	d_tanh = lambda x: 1 - np.tanh(x) ** 2

	d_relu = lambda x: 1 if x >0 else 0

	return {
		'sigmoid':d_sigmoid(x),
		'tanh':d_tanh(x),
		'relu': d_relu(x)
	}