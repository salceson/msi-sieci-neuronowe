# coding=utf-8
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork

from network import SupervisedNeuralNetwork

__author__ = 'Michał Ciołczyk'


class XORNeuralNetwork(SupervisedNeuralNetwork):
    """XOR Neural Network class.

    """
    def __init__(self, hidden_layer_size):
        """Creates XOR NN with specified hidden layer size.

        :param hidden_layer_size: Number of neurons in hidden layer.
        """
        data = SupervisedDataSet(2, 1)
        data.addSample([0, 0], [0])
        data.addSample([1, 1], [0])
        data.addSample([0, 1], [1])
        data.addSample([1, 0], [1])
        test_data = [([0, 1], 1), ([0.9, 0.2], 1)]
        SupervisedNeuralNetwork.__init__(self, data, test_data, test_data)
        self.network = buildNetwork(2, hidden_layer_size, 1, bias=True)


if __name__ == "__main__":
    xor_NN = XORNeuralNetwork(4)
    xor_NN.run(0.01, 0.99, 1000)