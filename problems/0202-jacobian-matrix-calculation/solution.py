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
    len_f = len(fx)
    len_x = len(x)
    
    # Initialize matrix with shape (len_f, len_x)
    jac = np.zeros((len_f, len_x))
    
    # Loop over each input variable j (columns)
    for j in range(len_x):
        x_perturbed = list(x)
        x_perturbed[j] += h
        
        # Derivative of all outputs with respect to input j
        df_dxj = (np.array(f(x_perturbed)) - fx) / h
        
        # Assign to column j
        jac[:, j] = df_dxj
        
    return jac.tolist()