import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    def batch_gd(X, y, weights, learning_rate, n_epochs):
        for i in range(n_epochs):
            preds = X @ weights
            loss = np.mean((y - preds) ** 2)

            # compute gradients
            grads = -2 * X.T @ (y - preds) / len(y)

            weights -= learning_rate * grads

        return weights
    
    def sgd(X, Y, weights, learning_rate, n_epochs):
        for i in range(n_epochs):
            for x, y in zip(X, Y):
                pred = np.dot(x, weights)
                loss = (y - pred) ** 2
                grad = x * -2 * (y - pred)
                weights -= learning_rate * grad
        return weights
    
    def minibatch_gd(X, y, weights, learning_rate, n_epochs, batch_size):
        for i in range(n_epochs):
            for j in range(0, len(X),batch_size):
                Xb  = X[j:j + batch_size]
                yb = y[j:j + batch_size]

                preds = Xb @ weights

                grads = -2 * Xb.T @ (yb - preds) / len(yb)

                weights -= learning_rate * grads
        return  weights

    if method == "batch":
        return batch_gd(X, y, weights, learning_rate, n_epochs)
    elif method == "stochastic":
        return sgd(X, y, weights, learning_rate, n_epochs)
    elif method == "mini_batch":
        return minibatch_gd(X, y, weights, learning_rate, n_epochs, batch_size)
    else:
        raise ValueError("Invalid method")

