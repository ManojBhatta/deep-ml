import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    indices = np.where(true_labels == 1)
    true_class_probs = predicted_probs[indices]
    true_class_probs = np.clip(true_class_probs, epsilon, 1.0)
    return  np.mean(-np.log(true_class_probs))