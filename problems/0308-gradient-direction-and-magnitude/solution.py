import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	norm = sum(x ** 2 for x in gradient)** (1/2)

	if norm == 0:
		direction = [0] * len(gradient)
		descent_dir = [0] * len(gradient)
	else:
		direction = [x/norm for x in gradient]
		descent_dir = [-x for x in direction]

	return {'magnitude':norm, 'direction':direction, 'descent_direction':descent_dir}