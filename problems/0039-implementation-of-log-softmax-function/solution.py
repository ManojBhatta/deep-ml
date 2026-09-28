import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	x = np.asarray(scores, dtype=np.float64)
	x -= x.max()
	return x - np.log(np.sum(np.exp(x)))