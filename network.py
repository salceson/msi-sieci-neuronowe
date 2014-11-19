# coding=utf-8

from matplotlib.pyplot import plot, show
from numpy import ndarray
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

__author__ = 'Michał Ciołczyk'


class SupervisedNeuralNetwork(object):
    """Abstract class for neural networks exercise

    """
    def __init__(self, learning_data, verify_data, test_data):
        """Creates NN with specified learning, verification and test data sets.

        :param learning_data: SupervisedDataSet containing learning data.
        :param verify_data: Set of tuples (x, y) containing verifying data.
        :param test_data: Set of tuples (x, y) containing test data.
        """
        self.learn_data = learning_data
        self.verify_data = verify_data
        self.test_data = test_data
        self.network = buildNetwork(1, 1, 1, bias=True)
        self.x = []
        self.err = []

    def learn(self, learning_rate, momentum, epochs, verbose=False, verbose_modulus=5):
        """Learns NN.

        :param learning_rate: NN learning rate
        :param momentum: NN momentum
        :param epochs: NN number of epochs
        :param verbose: if True, prints info about verification every verbose_modulus epochs
        :param verbose_modulus: rate to print info
        :return: PyBrain's NN class
        """
        if verbose:
            print "Training neural network..."
        trainer = BackpropTrainer(self.network, self.learn_data, learningrate=learning_rate, momentum=momentum)
        self.x = range(1, epochs + 1)
        for epoch in xrange(1, epochs + 1):
            to_print = verbose and epoch % verbose_modulus == 0
            if to_print:
                print "\tEpoch:", epoch, "/" + str(epochs)
            err = trainer.train()
            self.err.append(err)
        return self.network

    def verify(self, verbose):
        """Verifies NN.

        :rtype : float
        :param verbose: if True, prints info about verification
        :return: average error in verification data
        """
        return self.check_data(self.verify_data, "Verifying", verbose)

    def test(self):
        """Tests NN.

        :return: average error in test data
        """
        return self.check_data(self.test_data, "Testing", True)

    def check_data(self, data, text, verbose):
        """Performs tests on data on NN.

        :rtype : float
        :param data: data to test on
        :param text: text to insert in "... neural network"
        :param verbose: if True, prints info about tests
        :return: average error on data
        """
        if verbose:
            print "\t" + text + " neural network..."
            print "\t\tx\t\t\ty (actual)\t\ty(expected)"
        err = 0
        for x, y in data:
            if type(x) is not list:
                x = [x]
            if type(x[0]) is ndarray:
                x = x[0]
            y_act = self.network.activate(x)
            err += abs(y_act - y)
            if verbose:
                if type(x) is list and len(x) == 1:
                    x = x[0]
                if type(y_act) is ndarray and len(y_act) == 1:
                    y_act = y_act[0]
                print "\t\t" + str(x) + "\t\t" + str(y_act) + "\t\t" + str(y)
        err /= len(data)
        if verbose:
            if type(err) is ndarray and len(err) == 1:
                err = err[0]
            print "\tAverage error:", err
        return err

    def error_plot(self, to_plot=True, to_show=True):
        """Plots average error in function of epochs or returns errors.

        :param to_plot: specifies if we should call PyPlot's plot function
        :param to_show: specifies if we should call PyPlot's show function
        :return: (x, err) where x is set [1..epochs] and err is average_error(x)
        :raise ValueError: You need to first learn the NN before plotting.
        """
        if len(self.err) <= 0:
            raise ValueError("First learn the neural network")
        if to_plot:
            plot(self.x, self.err, '-')
            if to_show:
                show()
        return self.x, self.err

    def run(self, learning_rate, momentum, epochs, verbose=True, verbose_modulus=None):
        """Runs learning at specified parameters and then plots average_error(epoch) plot.

        :param learning_rate: NN learning rate
        :param momentum: NN momentum
        :param epochs: NN number of epochs
        :param verbose: if True, prints info about learning
        :param verbose_modulus: verbose_modulus: rate to print info
        """
        if verbose_modulus is None:
            verbose_modulus = int(epochs / 5) if epochs > 5 else 1
        self.learn(learning_rate, momentum, epochs, verbose, verbose_modulus)
        self.test()
        self.error_plot()
        pass

    def run_until_convergence(self, learning_rate, momentum, max_epochs, continue_epochs, verbose=True):
        """Runs learning at specified parameters until its convergence and then plots average_error(epoch) plot.

        :param max_epochs: maximum epochs to learn NN
        :param continue_epochs: after finding convergence, trainer will compute continue_epochs epochs to check for better minimum
        :param learning_rate: NN learning rate
        :param momentum: NN momentum
        :param verbose: if True, prints info about learning
        """
        self.learn_until_convergence(learning_rate, momentum, max_epochs, continue_epochs, verbose)
        self.test()
        self.error_plot()
        pass

    def learn_until_convergence(self, learning_rate, momentum, max_epochs, continue_epochs, verbose=True):
        if verbose:
            print "Training neural network..."
        trainer = BackpropTrainer(self.network, self.learn_data, learningrate=learning_rate, momentum=momentum)
        training_errors, validation_errors = trainer.trainUntilConvergence(continueEpochs=continue_epochs,
                                                                           maxEpochs=max_epochs)
        self.x = range(1, len(training_errors) + 1)
        self.err = training_errors
        return self.network