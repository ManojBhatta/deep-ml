import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	A = np.asarray(A, dtype=np.float64)
	m, n = A.shape

	q = np.zeros_like(A)
	r = np.zeros_like(A)
	# graham schmidt
	for col in range(n):
		v = A[:,col]
		for i in range(col):
			qi = q[:,i]
			r[i,col] = np.dot(qi, v)
			v -= np.dot(v,qi) * qi
		r[col,col] = np.linalg.norm(v)
		q[:,col] = v/np.linalg.norm(v)
	return (q, r)

