import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    x = x.astype(np.float64)
    numerical_grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        old_val = x[idx]

        x[idx] = old_val + epsilon
        fxph = f(x)

        x[idx] = old_val - epsilon
        fxmh = f(x)

        x[idx] = old_val

        numerical_grad[idx] = (fxph - fxmh) / (2 * epsilon)

    ng_norm = np.linalg.norm(numerical_grad)
    ag_norm = np.linalg.norm(analytical_grad)
    diff_norm = np.linalg.norm(numerical_grad - analytical_grad)

    den = ng_norm + ag_norm

    relative_error = diff_norm/den if den>0 else 0.0

    return (numerical_grad, relative_error)

