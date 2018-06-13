import sys
import numpy
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from random import shuffle
import math
from sklearn.metrics import classification_report,roc_curve

mu1 = [1,0]
mu2 = [0,1.5]

sigma1 = [[1,0.75],[0.75,1]]
sigma2 = [[1,0.75],[0.75,1]]

train_set0 = np.column_stack((np.random.multivariate_normal(mu1,sigma1,1500),np.zeros((1500,1))))
train_set1 = np.column_stack((np.random.multivariate_normal(mu2,sigma2,1500),np.ones((1500,1))))
train_dataset = np.concatenate((train_set0,train_set1))


test_set0 = np.column_stack((np.random.multivariate_normal(mu1,sigma1,500),np.zeros((500,1))))
test_set1 = np.column_stack((np.random.multivariate_normal(mu2,sigma2,500),np.ones((500,1))))
test_dataset= np.concatenate((test_set0,test_set1))

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def train(x, y, w, b, lr=0.1, L2_reg=0.00):
        #print(numpy.dot(x, w) + b)
        p_y_given_x = sigmoid(numpy.dot(x, w) + b)
        d_y = y - p_y_given_x
        
        w += lr * numpy.dot(x.T, d_y) - lr * L2_reg * w
        b += lr * numpy.mean(d_y, axis=0)
        
        return w,b


def negative_log_likelihood(x, y, w, b):
        sigmoid_activation = sigmoid(numpy.dot(x, w) + b)
        cross_entropy = - numpy.mean(
            numpy.sum(y * numpy.log(sigmoid_activation) +
            (1 - y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


def predict(x, w, b):
        return sigmoid(numpy.dot(x, w) + b)


def encode(y):
    out_data = []
    for i in y:
        out_data.append(int(i))
    out = numpy.zeros((len(y), 2))
    out[numpy.arange(len(y)), out_data] = 1
    return out


def fit(learning_rate=0.001, n_epochs=3000, n_in=2, n_out=2):

    x = train_dataset[:,[0,1]]
    y = encode(train_dataset[:,[2]])
    w = numpy.zeros((n_in,n_out))
    b = numpy.zeros(n_out)

    
    for epoch in range(n_epochs):
        w,b = train(x, y, w, b, lr=learning_rate, L2_reg=0.00)
        cost = negative_log_likelihood(x, y, w, b)
        #print(epoch, cost)
        learning_rate *= 0.95

    correct = 0
    predicted = []
    original = []
    for i in range(len(test_dataset[:,[0,1]])):
        temp = (predict(test_dataset[:,[0,1]][i], w, b))
        #print(np.argmax(temp, axis=0))
        crt = np.argmax(temp, axis=0)
        #print(crt,test_dataset[:,[2]][i])
        original.append(int(test_dataset[:,[2]][i]))
        predicted.append(crt)
        if(crt == test_dataset[:,[2]][i]):
            correct += 1
    return(original,predicted,(correct/len(test_dataset)))


one,two,acc= fit()
print("Accuracy: ", acc)
print(classification_report(two,one))
x,y,z = roc_curve(one,two)
plt.plot(x,y)
plt.show()
auc = np.trapz(x,y)
auc