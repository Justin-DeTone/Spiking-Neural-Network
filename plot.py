import matplotlib
import matplotlib.pyplot as plt
import numpy

class Plot:
    def __init__(self):
        self.avg_fire_rate = numpy.array([])
        self.run_number = numpy.array([])
        self.fig = None
        self.ax = None

    def plotPlot(self):
        self.fig, self.ax = plt.subplots()

        self.ax.plot(self.run_number, self.avg_fire_rate)
        self.ax.set_xlabel('Number of Runs')  # Add an x-label to the axes.
        self.ax.set_ylabel('Average Neuron Firing Frequency')  # Add a y-label to the axes.
        self.ax.set_title("Frequency of Firing Neurons as Network Runs")  # Add a title to the axes.

        plt.show()

    def addPoint(self, run, frequency):
        self.avg_fire_rate = numpy.append(self.avg_fire_rate, [frequency])
        self.run_number = numpy.append(self.run_number, [run])
