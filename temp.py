#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:23:52 2019

@author: rohrdr
"""

import ActivationFunctions as AF
import CostFunctions as CF
import DenseNetworks as DN

import numpy as np
import h5py


def load_data():
    """
    loading all data
    """

    # the training set
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    # the test set
    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])


    # classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    # reshaping dimensions
    train_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    # printing dimensions
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("\nSizes of sets")
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("\nShapes of training set")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_x's shape: " + str(train_x.shape))
    print ("train_set_y_orig shape: " + str(train_set_y_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("\nShapes of test set")
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_x's shape: " + str(test_x.shape))
    print ("test_set_y_orig shape: " + str(test_set_y_orig.shape))
    print ("test_y shape: " + str(test_y.shape))
    print ("\n\n\n")

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = load_data()
m_train = train_x.shape[1]
m_test = test_x.shape[1]

#
# 2 - layer network
#

# set random seed to 1 to compare with results of Andrew Ng
# when comparing the results, make sure that the random
# numbers in Layers line 39 are scaled by * 0.01
np.random.seed(1)


# create input for the neural net identical that of Andrew Ng
layers = []
layers.append([train_x.shape[0], 7, AF.ReLU()])
layers.append([7, 1, AF.Sigmoid()])

# initiate denseDN
myDN = DN.DenseNN(layers, CF.CrossEntropy(), learning_rate = 0.0075)

# train the neural net
myDN.train_dn(train_x, train_y, maxiter = 2500, print_frequency = 100)

Yhat = myDN.forward_propagation(train_x)
cost = myDN.lossFunc.get_loss(Yhat, train_y)

Yhatclass = np.where(Yhat > 0.5, 1, 0)

print ("The cost after training the network is " + str(cost))
print ("Accuracy = " + str(np.sum((Yhatclass == train_y) / m_train)))

# the test set
Yhat_test = myDN.forward_propagation(test_x)
cost_test = myDN.lossFunc.get_loss(Yhat_test, test_y)
Yhatclass_test = np.where(Yhat_test > 0.5, 1, 0)
print ("The cost of the test set is " + str(cost_test))
print ("Accuracy = " + str(np.sum((Yhatclass_test == test_y) / m_test)))
print ("\n\n\n")


#
# 4-layer network
#

# set random seed to 1 to compare with results of Andrew Ng
# when comparing the results, make sure that the random
# numbers in Layers line 39 are scaled by / np.sqrt(nx)
np.random.seed(1)
layers4 = []
layers4.append( [ train_x.shape[0], 20, AF.ReLU() ] )
layers4.append( [ 20, 7, AF.ReLU() ] )
layers4.append( [ 7, 5, AF.ReLU() ] )
layers4.append( [ 5, 1, AF.Sigmoid() ] )
myDN4 = DN.denseDN(layers4, CF.CrossEntropy(), learning_rate = 0.0075)
myDN4.train_dn(train_x, train_y, maxiter = 2500, print_frequency = 100)

Yhat = myDN4.forward_propagation(train_x)
cost = myDN4.lossFunc.get_loss(Yhat, train_y)

Yhatclass = np.where(Yhat > 0.5, 1, 0)

print ("The cost after training the network is " + str(cost))
print ("Accuracy = " + str(np.sum((Yhatclass == train_y) / m_train)))

# the test set
Yhat_test = myDN4.forward_propagation(test_x)
cost_test = myDN4.lossFunc.get_loss(Yhat_test, test_y)
Yhatclass_test = np.where(Yhat_test > 0.5, 1, 0)
print ("The cost of the test set is " + str(cost_test))
print ("Accuracy = " + str(np.sum((Yhatclass_test == test_y) / m_test)))
print ("\n\n\n")

