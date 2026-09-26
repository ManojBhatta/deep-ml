
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	dot = sum(a * b for a, b in zip(v, L))
	line_mag = (sum(x ** 2 for x in L))

	comp = dot/line_mag

	res = [comp * l for l in L]
	return  res

