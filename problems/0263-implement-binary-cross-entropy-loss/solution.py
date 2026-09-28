def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	import  numpy as np
	p = np.clip(y_pred, epsilon, 1-epsilon)
	y = np.asarray(y_true)

	return np.mean(-y*np.log(p) - (1-y)*np.log(1-p))