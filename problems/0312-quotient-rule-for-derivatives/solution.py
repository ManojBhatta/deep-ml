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
    def derivative(poly_coeffs: list, x:float):
        res = 0
        for i, c in enumerate(reversed(poly_coeffs)):
            res += c * i * x ** (i - 1) if i !=0 else 0
        return res

    def func(func_coeffs:list, x:float):
        res = 0
        for i, c in enumerate(reversed(func_coeffs)):
            res += c * x ** i
        return  res
    
    num = func(h_coeffs, x) * derivative(g_coeffs, x) - func(g_coeffs, x) * derivative(h_coeffs, x)
    den  = func(h_coeffs, x) ** 2
    if den == 0:
        return  -1
    return  num / den
