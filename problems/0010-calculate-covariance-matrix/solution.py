def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	if not vectors or not vectors[0]:
		return  []
	

	num_features = len(vectors)
	num_samples = len(vectors[0])

	dof = num_samples - 1 if num_samples > 1 else 1

	mean_centered = []
	for feature in vectors:
		mean = sum(feature) / num_samples
		mean_centered.append([x - mean for x in feature])
	
	cov = [[0] * num_features for _ in range(num_features)]

	for i in range(num_features):
		for j in range(i, num_features):
			cov_val = sum(a * b for a, b in zip(mean_centered[i], mean_centered[j])) / dof
			cov[i][j] = cov_val
			cov[j][i] = cov_val
	return  cov

