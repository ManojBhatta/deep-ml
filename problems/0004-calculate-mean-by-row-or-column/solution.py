def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	means = []
	if mode == 'row':
		for row in matrix:
			means.append(sum(row)/ len(row))
	elif mode == 'column':
		nr = len(matrix)
		nc = len(matrix[0])
		for i in range(nc):
			col = [row[i] for row in matrix]
			print(col)
			means.append(sum(col) / nr)
	return means