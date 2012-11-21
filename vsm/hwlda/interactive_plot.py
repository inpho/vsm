from matplotlib import pyplot

class InteractivePlot(object):

    def __init__(self, xlabel, ylabel):

        pyplot.ion()

        self.xlabel = xlabel
        self.ylabel = ylabel

        self.x = []
        self.y = []

    def update_plot(self, xval, yval):

        x = self.x
        y = self.y

        x.append(xval)
        y.append(yval)

        pyplot.clf()

        pyplot.plot(x, y, 'k')

        pyplot.xlabel(self.xlabel)
        pyplot.ylabel(self.ylabel)

        pyplot.draw()

if __name__ == '__main__':

    import time, math

    plt = InteractivePlot('x', 'y')

    for x in xrange(100):

        plt.update_plot(x, x * math.sin(0.4 * x))
        time.sleep(.01)
