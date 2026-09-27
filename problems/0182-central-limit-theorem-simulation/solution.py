import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    def get_run(dist:str):
        if dist == 'uniform':
            data = np.random.uniform(0, 1, size=(runs, n))
            return  (data - 0.5) / np.sqrt(1/12)
        elif dist == 'exponential':
            data =  np.random.exponential(1.0, size=(runs, n))
            return (data - 1)
        elif dist == 'bernoulli':
            data =  (np.random.rand(runs, n) < 0.3).astype(float)
            return  (data - 0.3 )/ np.sqrt(0.3*0.7)
        else:
            raise ValueError("invalid distribution")
    
    data = (get_run(distribution)* np.sqrt(n)).mean(axis=1)
    return {'mean':data.mean(), 'std':data.std()}