import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    def derivative(poly_coeffs, x):
        res = 0
        for i, c in enumerate(reversed(poly_coeffs)):
            if i != 0:
                res += c * i * x ** (i - 1)
        return res

    def evaluate(poly_coeffs, x):
        res = 0
        for i, c in enumerate(reversed(poly_coeffs)):
            res += c * x ** i
        return res

    g = evaluate(g_coeffs, x)
    h = evaluate(h_coeffs, x)

    if h == 0:
        return -1

    dg = derivative(g_coeffs, x)
    dh = derivative(h_coeffs, x)

    return (h * dg - g * dh) / h**2