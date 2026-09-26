import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""	
	A = np.array(A, dtype=np.float64)
	b = np.array(b, dtype=np.float64)
	n = len(b)
	
	Ab = np.hstack([A, b.reshape(-1, 1)])

	# convert to upper triangular
	for i in range(n):

		# get the row with max abs value in the column i:
		max_row = i + np.argmax(np.abs(Ab[i:, i]))

		if i != max_row:
			Ab[[i, max_row]] = Ab[[max_row, i]]

		# eliminate elements below pivot val for the column
		for j in range(i + 1, n):
			factor = Ab[j,i] / Ab[i, i]
			Ab[j, i:] -= factor * Ab[i, i:]

	
	sol = np.zeros_like(b)
	# backward substitution
	for i in range(n-1, -1, -1):
		sol[i] = (Ab[i,-1] - np.dot(Ab[i, i+1:n], sol[i+1:n])) / Ab[i, i]

	return  sol


