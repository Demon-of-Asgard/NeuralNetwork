#!/home/demon/anaconda3/bin/python3
#-------------------------------------------------------------------------------

import numpy as np
from pkg_Linked_List.linked_list import LinkedList
from pkg_Layer.layer import Layer

#-------------------------------------------------------------------------------

class NeuralNetwork(LinkedList):
    '''
    Neural nertwork constructor.
    __init__ args:
        nFeature:int --> #feature per training sample.
        Assumes that all training samples have same # fetures.

        nOut:int --> #classifications.

        nHiddenLayer:tuple(int, int) --> (#layer, #nodes per layer).
        Assumes all hidden layers has same # nodes. If we want to have hidden
        layers with different number of nodes, add_layer() method can be used.
    '''
    def __init__(self, nFeature:int, nOut:int, nHiddenLayer:tuple = (0,0)):
        LinkedList.__init__(self)
        # super(NeuralNetwork, self).__init__()
        self.nFeature = nFeature            #features per training sample.
        self.nOut = nOut                    # Number of classifications.
        self.nHiddenLayer = nHiddenLayer    # Number of the hiden layers.
        self.nLayer = 0                     # Total # layers.

        self.add_layer(self.nFeature, 0,'activation') # Adding activation layer.


        for i in range(self.nHiddenLayer[0]):   #Creating hidden layers
            self.add_layer(nHiddenLayer[1],self.nLayer, 'hidden')

        # Final layer
        self.add_layer(self.nOut, self.nLayer, 'final')

    def get_layer(self, loc:int = -1):
        '''Returns the layer obj at the given location. '''
        return self.get_link(loc)

    def add_layer(self, nNode, loc:int = -1, kind:str = 'hidden'):
        ''' Add a leyer to the nural network at the specified location. '''
        self.nLayer += 1
        if kind == 'activation':
            l = Layer(nNode, kind)
            l.initialize_activation()
            self.add_link(l, loc)
        elif kind == 'final':
            l = Layer(nNode, kind)
            l.initialize_activation()
            nNodeLast = self.get_link(loc-1).nNode
            l.initialize_params(nNodeLast)
            self.add_link(l, loc)
        else:
            l = Layer(nNode)
            l.initialize_activation()
            nNodeLast = self.get_link(loc-1).nNode
            l.initialize_params(nNodeLast)
            self.add_link(l, loc)

    def remove_layer(self, loc:int = -1):
        ''' Remove a leyer @loc from the NN. '''
        self.remove_link(loc)
        self.nLayer -= 1

    def load_training_values(self, xValue, yValue):
        self.get_layer(0).activation[1:,:] = xValue

    def train(self):
        pass

    def predict(self, feature):
        # Activating the input layer.
        self.get_layer(0).activation[1:,:] = feature

        # Propagating forward till the output layer.
        for i in range(1, self.nLayer):
            lLast = self.get_layer(i-1)
            l = self.get_layer(i)
            l.activate(lLast.activation)
            print('#layer: {} \nkind: {} \nactivation: \n{}'.
                format(i, l.kind, l.activation))
            print('theta: {}'.format(l.theta))

        # Returning the activation of the output layer as prediction.
        return self.get_layer(-1).activation

#-------------------------------------------------------------------------------
def main():
    import matplotlib.pyplot as plt
    xData = np.genfromtxt('./Test_Data/degitX.dat')
    ydata = np.genfromtxt('./Test_Data/degitY.dat')
    NN = NeuralNetwork(2,1)
    NN.get_layer(1).theta = np.array([-10, 20, 20])
    input = np.array([[0],[0]])
    print(NN.predict(input))
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print('\n'+'='*50)
    main()
    print('='*50+'\n')

#-------------------------------------------------------------------------------
