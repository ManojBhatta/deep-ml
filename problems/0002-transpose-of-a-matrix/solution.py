def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    m = len(a)
    n = len(a[0])
    a_t  = [[0]*m for _ in range(n)]

    for i,row in enumerate(a):
        for j, el in enumerate(row):
            a_t[j][i] = el
    return  a_t