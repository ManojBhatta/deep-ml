
import  numpy as np
def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.

	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences

	Returns:
		Jacobian matrix as list of lists
	"""
	fx = np.array(f(x))
	len_x = len(x)
	len_f = len(f(x))
	jac = np.zeros((len_f, len_x))

	for j in range(len_x):
		x_perturbed = x.copy()
		x_perturbed[j] += h
		dfxj = (np.array(f(x_perturbed)) - fx)/ h
		jac[:,j] = dfxj
	return  jac.tolist()

