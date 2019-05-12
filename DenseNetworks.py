#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:20:41 2019

@author: rohrdr
"""

import numpy as np
import abc
from Layers import Layer
from CostFunctions import LossFunction
from ActivationFunctions import ActivationFunction


class NeuralNetworks(abc.ABC):
    
    @abc.abstractmethod
    def __init__(self):
        
        return
    
    @abc.abstractmethod
    def _forward_propagation(self):
        
        vec = None
        
        return vec
    
    @abc.abstractmethod
    def _backward_propagation(self):
        
        return
    
    @abc.abstractmethod
    def _check_arrays(self, x):
    
        assert(isinstance(x, np.ndarray))
        assert((np.asarray(x).dtype.kind == "f") or (np.asarray(x).dtype.kind == "i"))
        
        return


class DenseNN(NeuralNetworks):
    """
    Neural Network with dense layers
    the class implements
        - the initialization of the network
        - the forward propagation
        - the backward propagation
        - the training of the network
        - the loss/cost function
    
    """
    
    def __init__(self, dimensions, loss_func, learning_rate=0.02):
        
        assert(isinstance(dimensions, list))
        assert(isinstance(loss_func, LossFunction))
        assert(isinstance(learning_rate, np.float))
        
        self.lossFunc = loss_func
        
        super().__init__()
        
        self.nlayers = len(dimensions)
        self.xdim = dimensions[0][0]
        self.ydim = dimensions[-1][1]
        self.learning_rate = learning_rate
        self.layers = []
        
        oldny = self.xdim
        
        # initialize all the layers
        for lay in dimensions:
            
            assert (len(lay) == 3)
            nx, ny, actfunc = lay
            assert(isinstance(nx, int))
            assert(oldny == nx)  # do the dimensions check?
            assert(isinstance(ny, int))
            assert(isinstance(actfunc, ActivationFunction))
            self.layers.append(Layer(nx, ny, actfunc))
            oldny = ny
        
        return
    
    def get_loss(self, x, y):
        """
        calculates the loss/cost with respect to y

        """
        
        self._check_arrays(x)
        self._check_arrays(y)
        assert(x.shape[0] == self.xdim)
        assert(y.shape[0] == self.ydim)
        assert(len(x.shape) >= 2)
        
        vec = self._forward_propagation(x)
        
        assert (vec.shape[0] == y.shape[0])
        
        loss = self.lossFunc.get_loss(vec, y)
        
        return loss
    
    def forward_propagation(self, x):
        """
        the forward propagation with feature vector x
        
        """
        
        self._check_arrays(x)
        assert(x.shape[0] == self.xdim)
        assert(len(x.shape) >= 2)
        
        vec = self._forward_propagation(x)
        
        return vec
    
    def _forward_propagation(self, x):
        
        super()._forward_propagation()

        vec = x
        for lay in self.layers:
            vec = lay.get_y(vec)
        
        return vec
    
    def backward_propagation(self, yhat, y):
        """
        the backward propagation with vectors yhat and y
        
        """
        
        self._check_arrays(yhat)
        self._check_arrays(y)
        
        d_ws, d_bs = self._backward_propagation(yhat, y)
        
        return d_ws, d_bs
    
    def _backward_propagation(self, yhat, y):
        
        super()._backward_propagation()
        
        vec = self.lossFunc.get_loss_der(yhat, y)
        
        d_ws = []
        d_bs = []
        
        self.layers.reverse()
        
        for lay in self.layers:
            vec, d_w, d_b = lay.get_grad(vec)
            d_ws.append(d_w)
            d_bs.append(d_b)
                
        self.layers.reverse()
        d_ws.reverse()
        d_bs.reverse()
        
        return d_ws, d_bs
    
    def train_dn(self, x, y, maxiter=20, print_frequency=10, print_flag=True):
        """
        train the network starting at x and labels y
        
        """
        
        self._check_arrays(y)
        self._check_arrays(x)
        assert(x.shape[0] == self.xdim)
        assert(len(x.shape) >= 2)
        assert(isinstance(print_frequency, np.int))
        
        for it in range(maxiter):
            yhat = self._forward_propagation(x)
            loss = self.lossFunc.get_loss(yhat, y)
            d_ws, d_bs = self._backward_propagation(yhat, y)
        
            for i, lay in enumerate(self.layers):
                lay.update_wb(-self.learning_rate * d_ws[i], -self.learning_rate * d_bs[i])
                
            if (it % print_frequency == 0) & print_flag:
                print("Iteration: " + str(it) + "   cost: " + str(loss))

        if print_flag:
            print("\n==================================")
            print("   FINAL ITERATION RESULTS")
            print("Iteration: " + str(it) + "   cost: " + str(loss) + "\n")
        
        return
    
    def get_wandb(self):
        
        ws = []
        bs = []
        
        for lay in self.layers:
            ws.append(lay.W)
            bs.append(lay.b)
        
        return ws, bs
    
    def _check_arrays(self, x):
        
        super()._check_arrays(x)
        
        return
