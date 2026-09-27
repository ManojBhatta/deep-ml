def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	trace = matrix[0][0] + matrix[1][1]
	det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
	eig1 = (trace + (trace**2 - 4*det)** 0.5) / 2
	eig2 = (trace - (trace**2 - 4*det)** 0.5) / 2
	eigenvalues = sorted([eig1, eig2], reverse=True)
	return eigenvalues