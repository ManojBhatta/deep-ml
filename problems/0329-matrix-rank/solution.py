import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    matrix = A.astype(np.float64, copy=True)
    m, n = matrix.shape

    row_idx = 0
    rank = 0
    
    for col_idx in range(n):
        # for complete pivoting get the max element of the submatrix
        submatrix = np.abs(matrix[row_idx:m, col_idx:n ])
        max_idx = np.unravel_index(np.argmax(submatrix), submatrix.shape)
        pivot_val = submatrix[max_idx]

        
        if pivot_val < tol:
            break
            
        pivot_row = row_idx +  max_idx[0]
        pivot_col = col_idx +  max_idx[1]

        # swap rows and columns
        if pivot_row != row_idx:
            matrix[[row_idx, pivot_row]] = matrix[[pivot_row, row_idx]]
        
        if pivot_col != pivot_col:
            matrix[:, [col_idx, pivot_col]] = matrix[:, [pivot_col,col_idx]]
        
        pivot = matrix[row_idx, col_idx]

        # eliminate entries of column below the pivot element

        for i in range(row_idx + 1, m):
            factor = matrix[i, col_idx] / pivot
            matrix[i,col_idx:] -= factor * matrix[row_idx, col_idx:]
            matrix[i, col_idx] = 0
        
        
        row_idx += 1
        rank += 1

        if row_idx >= m:
            break
    return rank
