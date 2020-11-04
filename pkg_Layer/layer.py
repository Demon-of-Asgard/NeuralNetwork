#-------------------------------------------------------------------------------
import numpy as np
from pkg_Transfer_Function.sigmoid import sigmoid
from pkg_Transfer_Function.sigmoid import d_sigmoid 

#-------------------------------------------------------------------------------
class Layer():
    def __init__(self, nNode:int = 1, kind:str = 'hidden'):
        self.nNode = nNode
        self.kind = kind
        self.theta = None

    def initialize_activation(self):
        if self.kind != 'final':
            self.activation = np.ones((self.nNode+1, 1))
        else:
            self.activation = np.zeros((self.nNode+1, 1))

    def initialize_params(self, nNodeLast:int = 1):
        self.__EPSILON__ =  10**(-4)
        self.nNodeLast = nNodeLast
        self.theta = (np.random.rand(self.nNode, self.nNodeLast+1)
                    *2*self.__EPSILON__ - self.__EPSILON__
                    ) # +1 is for bias term

    def activate(self, lastLayerActivation):
        if self.kind != 'final':
            self.net = np.dot(self.theta, lastLayerActivation)
            self.activation[1:,:] = sigmoid(self.net)
        else:
            self.net = np.dot(self.theta, lastLayerActivation)
            self.activation = sigmoid(self.net)

#-------------------------------------------------------------------------------
