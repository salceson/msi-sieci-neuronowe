# coding=utf-8

from pybrain import LinearLayer, SigmoidLayer, BiasUnit, FullConnection, FeedForwardNetwork
from network import SupervisedNeuralNetwork

__author__ = 'Michał Ciołczyk'


class PredictionNetwork(SupervisedNeuralNetwork):
    def __init__(self, data_provider, hidden_layer_size=5):
        self.data_provider = data_provider
        self.provided_data = self.data_provider.provide_learning_data()
        SupervisedNeuralNetwork.__init__(self, self.provided_data, self.provided_data, self.provided_data)
        self.network = self.build_network(hidden_layer_size)

    @staticmethod
    def build_network(hidden_layer_size):
        """Creates network

        :param hidden_layer_size: hidden layer size
        :return:
        """
        network = FeedForwardNetwork()
        # Create layers and bias unit
        input_layer = LinearLayer(5)
        hidden_layer = SigmoidLayer(hidden_layer_size)
        # output_layer = SigmoidLayer(1)
        output_layer = LinearLayer(1)
        bias = BiasUnit()
        # Create connections
        input_to_hidden = FullConnection(input_layer, hidden_layer)
        hidden_to_output = FullConnection(hidden_layer, output_layer)
        bias_to_hidden = FullConnection(bias, hidden_layer)
        bias_to_output = FullConnection(bias, output_layer)
        # Add layers to network
        network.addInputModule(input_layer)
        network.addModule(hidden_layer)
        network.addOutputModule(output_layer)
        network.addModule(bias)
        # Add connections to network
        network.addConnection(input_to_hidden)
        network.addConnection(hidden_to_output)
        network.addConnection(bias_to_hidden)
        network.addConnection(bias_to_output)
        # Sort modules and return
        network.sortModules()
        return network
