import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_der(x):
	return x*(1-x)

training_inputs=np.array([[0,0,0],
	[1,1,1],
	[1,0,1],
	[0,1,1]])

training_outputs=np.array([[0,1,1,0]]).T

np.random.seed(1)
synaptic_weights=2*np.random.random((3,1))-1
# print(synaptic_weights)

for i in range(1000000):
	input_layer=training_inputs
	output=sigmoid(np.dot(input_layer,synaptic_weights))

	error=training_outputs-output
	adjust=error*sigmoid_der(output)

	synaptic_weights+= np.dot(input_layer.T, adjust)
	
print(output)