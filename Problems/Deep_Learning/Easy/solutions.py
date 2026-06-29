import math

# Problem 22
def sigmoid(z: float) -> float:
	return 1 / (1 + math.pow(math.e, -z))