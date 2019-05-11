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
    def _forwardPropagation(self):
        
        vec = None
        
        return vec
    
    @abc.abstractmethod
    def _backwardPropagation(self):
        
        return
    
    @abc.abstractmethod
    def _check_arrays(self, X):
    
        assert( isinstance(X, np.ndarray) )
        assert( (np.asarray(X).dtype.kind == "f") or (np.asarray(X).dtype.kind == "i") )
        
        return

class denseNN(NeuralNetworks):
    """
    Neural Network with dense layers
    the class implements
        - the initialization of the network
        - the forward propagation
        - the backward propagation
        - the training of the network
        - the loss/cost function
    
    """
    
    def __init__(self, dimensions, lossFunc, learning_rate = 0.02):
        
        assert(isinstance(dimensions, list))
        assert(isinstance(lossFunc, LossFunction))
        assert(isinstance(learning_rate, np.float))
        
        self.lossFunc = lossFunc
        
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
            assert( isinstance( nx, int))
            assert( oldny == nx) # do the dimensions check?
            assert( isinstance( ny, int))
            assert( isinstance( actfunc, ActivationFunction) )
            self.layers.append(Layer(nx, ny, actfunc))
            oldny = ny
        
        return
    
    def get_loss(self, X, Y):
        """
        calculates the loss/cost with respect to Y

        """
        
        self._check_arrays(X)
        self._check_arrays(Y)
        assert( X.shape[0] == self.xdim)
        assert( Y.shape[0] == self.ydim)
        assert( len(X.shape) >=2)
        
        vec = self._forwardPropagation(X)
        
        assert (vec.shape[0] == Y.shape[0])
        
        loss = self.lossFunc.get_loss(vec, Y)
        
        return loss
    
    def forwardPropagation(self, X):
        """
        the forward propagation with feature vector X
        
        """
        
        self._check_arrays(X)
        assert( X.shape[0] == self.xdim)
        assert( len(X.shape) >=2)
        
        vec = self._forwardPropagation(X)
        
        return vec
    
    def _forwardPropagation(self, X):
        
        super()._forwardPropagation()

        vec = X
        for lay in self.layers:
            vec = lay.get_y(vec)
        
        return vec
    
    def backwardPropagation(self, Yhat, Y):
        """
        the backward propagation with vectors Yhat and Y
        
        """
        
        self._check_arrays(Yhat)
        self._check_arrays(Y)
        
        dWs, dbs = self._backwardPropagation(Yhat, Y)
        
        return dWs, dbs
    
    def _backwardPropagation(self, Yhat, Y):
        
        super()._backwardPropagation()
        
        vec = self.lossFunc.get_loss_der(Yhat, Y)
        
        dWs = []
        dbs = []
        
        self.layers.reverse()
        
        for lay in self.layers:
                vec, dW, db = lay.get_grad(vec)
                dWs.append(dW)
                dbs.append(db)
                
        self.layers.reverse()
        dWs.reverse()
        dbs.reverse()
        
        return dWs, dbs
    
    def trainDN(self, X, Y, maxiter = 20, print_frequency = 10):
        """
        train the network starting at X and labels Y
        
        """
        
        self._check_arrays(Y)
        self._check_arrays(X)
        assert( X.shape[0] == self.xdim)
        assert( len(X.shape) >=2)
        assert( isinstance(print_frequency, np.int))
        
        for iter in range(maxiter):
            Yhat = self._forwardPropagation(X)
            loss = self.lossFunc.get_loss(Yhat, Y)
            dWs, dbs = self._backwardPropagation(Yhat, Y)
        
            for i, lay in enumerate(self.layers):
                lay.update_wb(-self.learning_rate * dWs[i], -self.learning_rate * dbs[i])
                
            if ( iter%print_frequency == 0):
                print ("Iteration: " + str(iter) + "   cost: " + str(loss))
                
        print ("\n==================================")
        print ("   FINAL ITERATION RESULTS")        
        print ("Iteration: " + str(iter) + "   cost: " + str(loss) + "\n")
        
        return
    
    def get_Wandb(self):
        
        Ws = []
        bs = []
        
        for Lay in self.layers:
            Ws.append( Lay.W )
            bs.append( Lay.b )
        
        return Ws, bs
    
    def _check_arrays(self, X):
        
        super()._check_arrays(X)
        
        return
    
    
