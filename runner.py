# coding=utf-8

from numpy import std
from matplotlib import pyplot as plt

from data_provider import DataProvider
from prediction import PredictionNetwork


__author__ = 'Michał Ciołczyk'


def avg(ys):
    sum = 0
    for y in ys:
        sum += y
    return sum / len(ys)


if __name__ == '__main__':
    epochs = [50, 100, 200]
    learning_rates = [0.01, 0.1, 0.2]
    momentums = [0.5, 0.75, 0.99]
    hidden_layer_size = 30
    repeats = 10
    normalize = True
    denormalize = True

    dp = DataProvider('data2.csv', 100, normalize)
    test_data = dp.provide_test_data()

    for e in epochs:
        for lr in learning_rates:
            for m in momentums:
                print 'Epochs:', e, 'Learning rate:', lr, 'Momentum:', m

                networks = []

                for i in xrange(0, repeats):
                    pn = PredictionNetwork(dp, hidden_layer_size)
                    pn.learn(lr, m, e, False)
                    networks.append(pn.network)

                days = range(1, len(test_data) + 1)
                real_vals = []
                stddevs = []
                avgs = []

                for (x, y) in test_data:
                    yc = y[0]
                    ys = map(lambda n: n.activate(x)[0], networks)
                    err = map(lambda yp: abs(yp - yc), ys)
                    if denormalize:
                        yc = dp.max * yc + dp.min
                        ys = map(lambda a: dp.max * a + dp.min, ys)
                        err = map(lambda a: dp.max * a, err)
                    stddev = std(err)
                    avgval = avg(ys)
                    real_vals.append(yc)
                    stddevs.append(stddev)
                    avgs.append(avgval)

                plt.figure()
                plt.plot(days, real_vals)
                plt.errorbar(days, avgs, stddevs)
                plt.savefig('error_plots/' + str(e) + '_' + str(lr) + '_' + str(m) + '.png')
                plt.close()