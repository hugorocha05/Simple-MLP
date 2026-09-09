"""
This code implements 3 classes and 2 helper functions that can be used to create/define a simple MLP/NN with only forward pass.
It will serve as a base for other code that will eventually implement a way to train said MLP/NN (through Neuroevolution, Backpropagation etc)
"""

import random
import math

class Neuron:
	def __init__(self, weights, bias) -> None:
		self.weights = weights
		self.bias = bias

	def forward(self, inputs):  # Simple weighted sum of inputs x their respective weights, plus a bias
		if len(inputs) != len(self.weights):
			raise ValueError("Number of inputs must match number of weights. If you created a MLP with the create_mlp_from_matrix() function, please verify that the mlp_matrix argument is structured correctly.")
		
		weighted_sum = 0

		for input_value, weight in zip(inputs, self.weights):
			weighted_sum += input_value * weight

		weighted_sum += self.bias

		return weighted_sum

class Layer:
	def __init__(self, neurons, activation_type) -> None:  # A layer is a group of neuron with the same inputs and activation function
		self.neurons = neurons
		self.activation_type = activation_type

	def forward(self, inputs):  # Each neurons weighted_sum is passed through the activation function and then added to the outputs vector, which will ahve len = num of neurons in said layer
		layer_outputs = []

		for neuron in self.neurons:
			weighted_sum =  neuron.forward(inputs)

			neuron_output = self.activation(weighted_sum)

			layer_outputs.append(neuron_output)

		return layer_outputs

	def activation(self, weighted_sum):  # Different types of activation function that are regularly used, research to know more
		if self.activation_type == 'relu':
			return max(0, weighted_sum)

		elif self.activation_type == 'sigmoid':
			return 1 / (1 + math.exp(-weighted_sum))

		elif self.activation_type == 'tanh':
			return math.tanh(weighted_sum)

		elif self.activation_type == 'linear step':
			return 1 if weighted_sum >= 0 else 0
		
		else:
			raise ValueError(f"Unknown activation type: {self.activation_type}")
		
class MLP:  # A structured group of layers that work sequentially, with the output of one being the input of the next
	def __init__(self, layers) -> None:
		self.layers = layers

	def predict(self, inputs):
		current_inputs = inputs

		for layer in self.layers:
			layer_output = layer.forward(current_inputs)

			current_inputs = layer_output

		return layer_output

	def randomize_params(self, param_min, param_max):
		for layer in self.layers:
			for neuron in layer.neurons:
				new_weights = []

				for _ in range(len(neuron.weights)):
					new_weights.append(random.uniform(param_min, param_max))

				neuron.weights = new_weights[:]
				neuron.bias = random.uniform(param_min, param_max)

	def set_params(self, parameter_matrix):
		if len(parameter_matrix) != len(self.layers):
			raise ValueError("Parameter Matrix length must match up with the number of Layers in the MLP")
		
		for layer_info, layer in zip(parameter_matrix, self.layers):

			if len(layer_info) != len(layer.neurons):
				raise ValueError("Layer info length must match up with the number of Neurons in the Layer")

			for neuron_info, neuron in zip(layer_info, layer.neurons):

				if len(neuron_info) != len(neuron.weights) + 1:
					raise ValueError("Neuron info length must match up with the number of weights plus one bias in the Neuron")
				
				num_weights = len(neuron.weights)

				neuron.weights = neuron_info[:num_weights]
				neuron.bias = neuron_info[num_weights]

		# Parameter Matrix Example ----------------------------

		"""
		Each row represents a layer of the MLP
		The last value of each vector in each row (in this case the 3 from the first vector and the 4 from the second vector of the first row and the 5 of the only vector in the second row) represent the bias of a single neuron and the rest of the values before it represent the weights of that same neuron
		In this example we have 2 layers and 3 neuron (2 in the first and 1 in the second), the first one has weights [1, 2] and bias 3, the second one has weights [2, 3] and bias 4, and the third has weights [3, 4] and bias 5

		param_matrix = [
			[[1, 2, 3], [2, 3, 4]],
			[[3, 4, 5]]
		]

		"""
		# -------------------------------------------------
					


def create_architecture_only_mlp(architecture):  # Creates a MLP with the correct structure/architecture but all weights and biases zeroed
	layers_in_mlp = []

	num_inputs = architecture[0]

	for layer_info in architecture[1:]:
		neurons_in_layer = []

		num_neurons = layer_info[0]
		activation = layer_info[1]

		for _ in range(num_neurons):
			neurons_in_layer.append(Neuron([0 for _ in range(num_inputs)], 0))

		layers_in_mlp.append(Layer(neurons_in_layer, activation))

		num_inputs = num_neurons  # This is because the number of inputs of each Neuron in layer X is the same as the number of total Neurons in Layer X-1

	return MLP(layers_in_mlp)

	# Architecture Example --------------------------

	"""
	Number of inputs of the MLP
	[Number of Neurons, Activation function] for each layer - including the output layer

	In this case there are 2 inputs, 2 Neurons in a hidden layer and 1 output Neuron

	architecture = [
		2,
		[2, 'relu'],
		[1, 'sigmoid']
	]

	"""
	# -----------------------------------------------

def create_mlp_from_matrix(mlp_matrix):
	layers_in_mlp = []

	for layer_info in mlp_matrix:
		neurons_in_layer = []

		activation = layer_info[-1]

		for neuron_info in layer_info[:-1]:
			neurons_in_layer.append(Neuron(neuron_info[:-1], neuron_info[-1]))

		layers_in_mlp.append(Layer(neurons_in_layer, activation))

	return MLP(layers_in_mlp)

	# MLP Matrix Example --------------------------------------------

	"""
	Just like the parameter matrix except the activation function of each layer is added to the end
	with this, you can create a specific MLP with one command and one data structure, instead of relying on creating a MLP with only architecture and then setting the parameters

	mlp_matrix = [
		[[1, 2, 3], [2, 3, 4], 'relu'],
		[[3, 4, 5], 'sigmoid']
	]
	
	"""


# Example usages ----------------------------------------------------------------------------------

inputs = [1, 2]

# Architecture + setting parametrs manually

architecture = [
	2,
	[2, 'relu'],
	[1, 'sigmoid']
]

mlp1 = create_architecture_only_mlp(architecture)

param_matrix = [
	[[1, 2, 3], [2, 3, 4]],
	[[3, 4, 5]]
]

mlp1.set_params(param_matrix)

print(mlp1.predict(inputs))

# ---------------------------------------------------------

# Architecture + randomizing parameters

architecture = [
	2,
	[2, 'relu'],
	[1, 'sigmoid']
]

mlp2 = create_architecture_only_mlp(architecture)

mlp2.randomize_params(-1, 1)

print(mlp2.predict(inputs))

# ---------------------------------------------------------

# MLP Matrix (with implicit architecture)

mlp_matrix = [
	[[1, 2, 3], [2, 3, 4], 'relu'],
	[[3, 4, 5], 'sigmoid']
]

mlp3 = create_mlp_from_matrix(mlp_matrix)

print(mlp3.predict(inputs))
# -------------------------------------------------------------------------------------------------

