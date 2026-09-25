import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    
    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    if norm_type == "l1":
        return np.sum(np.abs(arr)).astype(np.float32)
    elif norm_type == "l2":
        return np.sqrt(np.sum(arr ** 2)).astype(np.float32)
    elif norm_type == "linf":
        return np.max(np.abs(arr)).astype(np.float32)
    elif norm_type == "frobenius":
        if len(arr.shape) != 2:
            raise ValueError("arr must be 2d")
        else:
            return np.sqrt(np.sum(arr ** 2)).astype(np.float32)
        

    