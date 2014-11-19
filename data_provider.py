# coding=utf-8

from pybrain.datasets import SupervisedDataSet

__author__ = 'Michał Ciołczyk'


class DataProvider(object):
    """Data provider for prediction network.

    """

    def __init__(self, filename, max_count=50000000, normalize=True):
        """Data provider for prediction network.

        :param max_count: max
        :param filename: file to read from
        """
        self.filename = filename
        self.max = 1
        self.min = 0
        self.data = SupervisedDataSet(5, 1)
        self.test_data = []
        read_data = []

        with open(self.filename) as f:
            f.readline()
            line = f.readline()
            while line:
                date, value = line.split(';')
                read_data.append(float(value))
                line = f.readline()

        learn_data = read_data[:min(max_count, len(read_data))]
        read_data = read_data[min(max_count, len(read_data)):min(max_count, len(read_data))+100]

        search_data = learn_data + read_data

        if normalize:
            self.min = min(search_data)
            search_data = map(lambda x: x - self.min, search_data)
            self.max = max(search_data)

        learn_data = map(lambda x: (x-self.min)/self.max, learn_data)
        read_data = map(lambda x: (x-self.min)/self.max, read_data)

        for i in xrange(5, len(learn_data) - 1):
            row = learn_data[i - 5:i]
            self.data.addSample(row, [learn_data[i]])

        for i in xrange(5, len(read_data) - 1):
            row = learn_data[i - 5:i]
            self.test_data.append((row, [read_data[i]]))
        pass

    def provide_learning_data(self):
        """Provides learning data set.

        :return: learning data set
        """
        return self.data

    def provide_test_data(self):
        """Provides test data set.

        :return: test data set
        """
        return self.test_data
