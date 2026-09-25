def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	# Your code here
	import  numpy as np
	mat = np.array(matrix)
	det = np.linalg.det(mat)
	trace = sum(mat[i,i] for i in range(mat.shape[0]))
	return (det, trace)