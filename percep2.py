import numpy as np

class NeuralNetwork():
	def __init__(self):
		np.random.seed(1)
		self.synaptic_weights=2*np.random.random((3,1))-1

	def sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def sigmoid_der(self,x):
		return x*(1-x)

	def train(self, training_inputs, training_outputs, training_iterations):
		for i in range(training_iterations):	
			output=self.think(training_inputs)
			error=training_outputs - output
			adjust=np.dot(training_inputs.T, error*self.sigmoid_der(output))
			self.synaptic_weights += adjust

	def think(self,inputs):
		inputs=inputs.astype(float)
		output=self.sigmoid(np.dot(inputs, self.synaptic_weights))
		return output

if __name__=="__main__":
	neural_network=NeuralNetwork()
	# print("random weights")
	# print(neural_network.synaptic_weights)

	training_inputs=np.array([[0,1,0],
	[0,0,1],
	[1,0,0],
	[1,1,0],
	[1,1,1]
	])

	training_outputs=np.array([[1,0,0,1,1]]).T

	neural_network.train(training_inputs, training_outputs, 10000)
	# print("after weights")
	# print(neural_network.synaptic_weights)

	A=str(input("input is: "))
	B=str(input("input is: "))
	C=str(input("input is: "))

	print(neural_network.think(np.array([A, B, C])))