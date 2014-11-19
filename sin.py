# coding=utf-8
import math

from numpy import arange
from pybrain import SigmoidLayer, LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit
from pybrain.datasets import SupervisedDataSet

from network import SupervisedNeuralNetwork


__author__ = 'Michał Ciołczyk'


class SinNeuralNetwork(SupervisedNeuralNetwork):
    def __init__(self, hidden_layer_size):
        """Sin Neural Network class.

        :param hidden_layer_size: Number of neurons in hidden layer.
        """
        learning_data = SinNeuralNetwork.create_sin_learning_sample(-math.pi / 2, math.pi / 2, math.pi / 10)
        test_data = SinNeuralNetwork.create_sin_sample(-math.pi, math.pi, math.pi / 20)
        verify_data = SinNeuralNetwork.create_sin_sample(-math.pi / 2, math.pi / 2, math.pi / 20)
        SupervisedNeuralNetwork.__init__(self, learning_data, verify_data, test_data)
        self.network = SinNeuralNetwork.create_network(hidden_layer_size)

    @staticmethod
    def create_sin_learning_sample(start_range, end_range, step):
        """Samples sin function as SupervisedDataSet.

        :rtype : SupervisedDataSet
        :param start_range: start of sampling range (inclusive)
        :param end_range: end of sampling range (exclusive)
        :param step: sampling step
        :return: sampled data
        """
        data = SupervisedDataSet(1, 1)
        for (x, y) in SinNeuralNetwork.create_sin_sample(start_range, end_range, step):
            data.addSample([x], [y])
        return data

    @staticmethod
    def create_sin_sample(start_range, end_range, step):
        """Samples sin function as list.

        :rtype : list
        :param start_range: start of sampling range (inclusive)
        :param end_range: end of sampling range (exclusive)
        :param step: sampling step
        :return: sampled data
        """
        data = []
        for x in arange(start_range, end_range, step):
            data.append((x, math.sin(x)))
        return data

    @staticmethod
    def create_network(hidden_layer_size):
        """Creates sin NN.

        :param hidden_layer_size: size of hidden layer
        :return: network
        """
        # Create network
        network = FeedForwardNetwork()
        # Create network layers and bias
        input_layer = LinearLayer(1)
        hidden_layer = SigmoidLayer(hidden_layer_size)
        output_layer = SigmoidLayer(1)
        bias = BiasUnit()
        # Create connections
        input_hidden_connection = FullConnection(input_layer, hidden_layer)
        hidden_output_connection = FullConnection(hidden_layer, output_layer)
        bias_hidden_connection = FullConnection(bias, hidden_layer)
        bias_output_connection = FullConnection(bias, output_layer)
        # Add network layers and bias to network
        network.addInputModule(input_layer)
        network.addModule(hidden_layer)
        network.addOutputModule(output_layer)
        network.addModule(bias)
        # Add connections between layers and bias
        network.addConnection(input_hidden_connection)
        network.addConnection(hidden_output_connection)
        network.addConnection(bias_hidden_connection)
        network.addConnection(bias_output_connection)
        # Sort modules and return network
        network.sortModules()
        return network

if __name__ == "__main__":
    sin_NN = SinNeuralNetwork(10)
    # sin_NN.run(0.01, 0.99, 1000)
    sin_NN.run_until_convergence(0.01, 0.99, 10000, 50, True)