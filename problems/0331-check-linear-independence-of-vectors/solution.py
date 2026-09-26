import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # create a matrix from the vectors
    if len(vectors) == 0:
        return True
    
    matrix = np.array(vectors)
    num_vectors, dim = matrix.shape

    if num_vectors > dim:
        return  False

    rank = np.linalg.matrix_rank(matrix)

    if rank == num_vectors:
        return  True
    else:
        return False