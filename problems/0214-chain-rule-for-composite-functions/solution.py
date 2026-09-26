import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	def derivative(func_name, value):
		if func_name == "square":
			return (2 * value, value ** 2)
		elif func_name == 'sin':
			return (np.cos(value), np.sin(value))
		elif func_name == 'exp':
			return (np.exp(value), np.exp(value))
		elif func_name == 'log':
			return (1 / value , np.log(value))
		else:
			raise ValueError("Invalid func name")
	res = 1

	for func in reversed(functions):
		df, f = derivative(func, x)
		print(f'func{func}, value {x} derivative {df}')
		res *= df
		x = f
	return  res

