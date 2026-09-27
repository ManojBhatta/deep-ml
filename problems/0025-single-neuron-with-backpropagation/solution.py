import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here

	b, l = features.shape
	labels.reshape(1, -1) # into a row vector

	def forward(X, weights, biases):
		# X -> (b, l) # W -> (l, ) # bias -> (l, )
		out =   X @ weights + biases #(l, )
		# sigmoid
		return  1 / (1 + np.exp(-out))

	def sigmoid(x):
		return  1 / (1 + np.exp(-x))
	
	weights = initial_weights
	biases = initial_bias
	mse_values = []

	for i in range(epochs):
		# forward
		out = forward(features,weights, biases)

		# compute loss
		loss = np.mean((labels - out)**2)
		mse_values.append(loss)

		# compute gradients
		d_out = -2 * (labels - out) / len(labels)
		d_sigmoid = d_out * out * ( 1  - out)
		d_bias = d_sigmoid.sum()
		d_weights =  features.T @ d_sigmoid

		# update
		weights -= learning_rate * d_weights
		biases -= learning_rate * d_bias


	# return updated_weights, updated_bias, mse_values
	return weights, biases, mse_values