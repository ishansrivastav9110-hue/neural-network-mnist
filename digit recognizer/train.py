from model import NeuralNetwork
import numpy as np
import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
mnist=fetch_openml("mnist_784",version=1,as_frame=False)
x=mnist.data.astype(np.float32)
y=mnist.target.astype(np.int64)
x=x/255.0
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,stratify=y
)
def target_array(idx):
    target=np.zeros(10)
    target[idx]=1
    return target
network=NeuralNetwork()
learning_rate=0.01
epochs=15
for i in range(epochs):
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
    total_loss=0
    for j in range(len(x_train)):
        target=target_array(y_train[j])
        forward=network.forward(x_train[j])
        loss=-np.sum(target*np.log(forward+1e-12))
        total_loss+=loss
        network.backprop(target,learning_rate)
    avg_loss=total_loss/len(x_train)
    print(f"loss after epoch{i+1}={avg_loss: .4f}")
    learning_rate *=0.96
predictions=[]
for i in range(len(x_test)):
    output=network.forward(x_test[i])
    prediction=np.argmax(output)
    predictions.append(prediction)
accuracy=accuracy_score(y_test,predictions)
print(f"Accuracy: {accuracy*100}")
with open("trained_network1.pkl", "wb") as f:
    pickle.dump(network, f)
