import numpy as np
class Layer():
    def __init__(self,num_inputs,num_neurons):
        self.weights=np.random.randn(num_neurons,num_inputs)*np.sqrt(2/num_inputs)
        self.biases=np.zeros(num_neurons)
    def relu(self,x):
        return np.maximum(0,x)
    def relu_derivative(self,x):
        return (x>0).astype(float)
    def forward(self,inputs):
        self.inputs=inputs
        self.z=np.dot(self.weights,inputs)+self.biases
        self.output=self.relu(self.z)
        return self.output
    def backprop(self,grad,learning_rate):
        delta=grad*self.relu_derivative(self.z)
        prev_grad=self.weights.T@delta
        weight_grad=np.outer(delta,self.inputs)
        bias_grad=delta
        self.weights-=learning_rate*weight_grad
        self.biases-=learning_rate*bias_grad
        return prev_grad
class Output(Layer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons)
    def softmax(self,x):
        exp_x=np.exp(x-np.max(x))
        return exp_x/np.sum(exp_x)
    def forward(self,inputs):
        self.inputs=inputs
        z=np.dot(self.weights,inputs)+self.biases
        self.output=self.softmax(z)
        return self.output
    def backprop(self,target,learning_rate):
        error=self.output-target
        weight_grad=np.outer(error,self.inputs)
        prev_grad=self.weights.T@error
        self.weights-=learning_rate*weight_grad
        self.biases-=learning_rate*error
        return prev_grad

class NeuralNetwork():
    def __init__(self):
        self.hidden1=Layer(784,256)
        self.hidden2=Layer(256,128)
        self.output=Output(128,10)
    def forward(self,inputs):
        hidden1_output=self.hidden1.forward(inputs)
        hidden2_output=self.hidden2.forward(hidden1_output)
        output=self.output.forward(hidden2_output)
        return output
    def backprop(self,target,learning_rate):
        grad=self.output.backprop(target,learning_rate)
        grad1=self.hidden2.backprop(grad,learning_rate)
        self.hidden1.backprop(grad1,learning_rate)
def target_array(idx):
    target=np.zeros(10)
    target[idx]=1
    return target