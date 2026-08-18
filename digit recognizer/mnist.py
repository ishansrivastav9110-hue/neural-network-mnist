from sklearn.datasets import  fetch_openml
import random
from sklearn.model_selection import train_test_split
import numpy as np
import math
from sklearn.metrics import accuracy_score

mnist=fetch_openml("mnist_784",version=1,as_frame=False)
x=mnist.data.astype(np.float32)
y=mnist.target.astype(np.int64)
x=x/255.0
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)
class Layer():
    def __init__(self,num_inputs,num_neurons):
        self.weights=np.random.randn(num_neurons,num_inputs)*0.01
        self.biases=np.zeros(num_neurons)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def forward(self,inputs):
        self.inputs=inputs
        self.z=np.dot(self.weights,inputs)+self.biases
        self.output=self.sigmoid(self.z)
        return self.output
    def backprop(self,grad,learning_rate):
        delta=grad*self.output*(1-self.output)
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
        self.hidden=Layer(784,128)
        self.output=Output(128,10)
    def forward(self,inputs):
        hidden_output=self.hidden.forward(inputs)
        output=self.output.forward(hidden_output)
        return output
    def backprop(self,target,learning_rate):
        grad=self.output.backprop(target,learning_rate)
        self.hidden.backprop(grad,learning_rate)
def target_array(idx):
    target=np.zeros(10)
    target[idx]=1
    return target
network=NeuralNetwork()
learning_rate=0.01
epochs=15
for i in range(epochs):
    for j in range(len(x_train)):
        target=target_array(y_train[j])
        forward=network.forward(x_train[j])
        network.backprop(target,learning_rate)
predictions=[]
for i in range(len(x_test)):
    output=network.forward(x_test[i])
    prediction=np.argmax(output)
    predictions.append(prediction)
accuracy=accuracy_score(y_test,predictions)
print(f"Accuracy: {accuracy*100}")
