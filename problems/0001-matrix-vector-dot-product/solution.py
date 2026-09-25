def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	# matrices shape
	rows = len(a)
	cols = len(a[0])

	l = len(b)

	if cols != l:
		return  -1

	res = [0]*rows
	for i, row in enumerate((a)):
		for j, col in enumerate((row)):
			res[i] += col * b[j]
	return res
			