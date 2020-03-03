import numpy as np
import math
import random


class Neuron:
    def __init__(self, preSN=None, preWeight=None, selfWeight=0.5):
        if preSN is None:
            self.preSN = []
        if preWeight is None:
            self.preWeight = []
        self.postWeight = selfWeight
        self.postSN = 0
        self.weight_training = []
        for weight in self.preWeight:
            self.weight_training.append(0)

        self.timer = 0
        self.pre_time = []
        for weight in self.preWeight:
            self.pre_time.append(10000)
        self.current_time = 10000

        self.firingWin = []
        self.firingAvg = 0
        self.isAvg = False

        self.time_1 = 1
        self.time_2 = 1
        self.voltage = 0
        self.bias = 0
        self.sSum_all = []
        self.revSSum = 0

        self.time_constant = 1
        self.resistance = 3
        self.voltage_threshold = 3
        self.delta_time = 0.05

    def addPreSN(self, neuronClass, weight=None, ):
        if weight is None:
            self.preWeight.append(np.random.normal(0, 5))
        self.preSN.append(neuronClass)
        self.sSum_all.append(0)
        self.weight_training.append(0)
        self.pre_time.append(10000)

    def prunePreSN(self, threshold):
        for idx, value in enumerate(self.preWeight):
            if abs(value) < threshold:
                del self.preWeight[value]
                del self.preSN[value]
                del self.sSum_all[value]
                idx -= 1   # Check that this works!
            pass

    def getSSum(self, time_constant, idxOfPreSN):
        # Also updates S value for neuron at specified index
        tmp = math.exp(-1.0/time_constant) * self.sSum_all[idxOfPreSN] + self.preSN[idxOfPreSN].postSN
        self.sSum_all[idxOfPreSN] = tmp
        return tmp

    def getRevSSum(self, time_constant):
        # Also updates S value for neuron at specified index
        tmp = -math.exp(-1.0/time_constant)*self.revSSum - self.postSN
        self.revSSum = tmp
        return tmp

    def calcCurrent(self):
        sum = 0
        for idx, neuron in enumerate(self.preSN):
            sum += self.preWeight[idx] * self.getSSum(self.time_1, idx)
        sum += self.postWeight * self.getRevSSum(self.time_2)
        sum += self.bias
        return sum

    def doesFire(self):
        self.voltage += (-self.voltage + self.resistance * self.calcCurrent()) * self.delta_time / float(self.time_constant)
        if self.voltage >= self.voltage_threshold:
            self.voltage = 0
            self.postSN = 1
            return 1
        self.postSN = 0
        return 0

    def STDPfunction(self, time_constant, x, shift):
        if x - shift >= 0:
            return math.exp((-x+shift)/time_constant)
        else:
            return -math.exp((-x+shift)/time_constant)

    def returnTime(self, count):
        return count * self.delta_time

    def resetTimeValues(self):
        for time in self.pre_time:
            time = time - 1000
        self.current_time = self.current_time - 1000
        self.timer = self.timer - 1000

    def calcWeightUpdate(self):
        for idx2, neuron in enumerate(self.preSN):
            if neuron.postSN == 1:
                self.pre_time[idx2] = self.timer
                if self.current_time < 10000:
                    self.weight_training[idx2] += self.STDPfunction(1, self.returnTime(self.current_time - self.pre_time[idx2]), 0)
        if self.postSN == 1:
            self.current_time = self.timer
            for idx2, neuron in enumerate(self.preSN):
                if self.pre_time[idx2] < 10000:
                    self.weight_training[idx2] += self.STDPfunction(1, self.returnTime(self.current_time - self.pre_time[idx2]), 0)

    def updateWeights(self, count):
        self.calcWeightUpdate()
        self.timer += 1
        if self.timer > 500:
            self.resetTimeValues()
        if self.timer % count == 0:
            for idx2, preWeight in enumerate(self.preWeight):
                preWeight += self.weight_training[idx2]

    def updateFRAvg(self, window):
        if self.timer < window and not self.isAvg:
            self.firingWin.append(self.postSN)
        elif self.timer == window and not self.isAvg:
            self.firingWin.append(self.postSN)
            self.isAvg = True
            for value in self.firingWin:
                self.firingAvg += value
            self.firingAvg = self.firingAvg / float(window)
        else:
            x = self.firingWin[0]
            del self.firingWin[0]
            self.firingWin.append(self.postSN)
            if x == 1:
                self.firingAvg -= 1.0 / window
            self.firingAvg += self.postSN / float(window)


class Input:
    def __init__(self, isRandom = False):
        self.isRandom = isRandom
        self.input = 0.5
        if isRandom:
            self.input = random.random()
        self.postSN = 0

        self.sSum = 0
        self.revSSum = 0
        self.time_1 = 1
        self.time_2 = 1
        self.time_constant = 0.5

        self.preWeight = 2
        self.postWeight = 0.3

        self.voltage = 0
        self.resistance = 3
        self.voltage_threshold = 1
        self.delta_time = 0.05

    def getSSum(self, time_constant):
        # Also updates S value
        self.sSum = math.exp(-1.0/time_constant) * self.sSum + self.input
        return self.sSum

    def getRevSSum(self, time_constant):
        self.revSSum = - math.exp(-1.0 / time_constant) * self.revSSum - self.postSN
        return self.revSSum

    def calcCurrent(self):
        return self.preWeight * self.getSSum(self.time_1) + self.postWeight * self.getRevSSum(self.time_2)

    def doesFire(self):
        self.voltage += (self.resistance * self.calcCurrent()) * self.delta_time / float(self.time_constant)
        if self.voltage >= self.voltage_threshold:
            self.voltage = 0
            self.postSN = 1
            return 1
        self.postSN = 0
        return 0


def setupFeedForward(*layers):
    # Sets up a feedforward network with random inputs and with neuron layers of length given by list of ints
    if len(layers) < 2:
        print("Not enough layers")
    else:
        neurons = [[]]
        for idx in range(layers[0]):
            neurons[0].append(Input(True))
        for layer_num in range(len(layers) - 1):
            neurons.append([])
            for nPerLayer in range(layers[layer_num + 1]):
                tmpNeuron = Neuron()
                neurons[layer_num + 1].append(tmpNeuron)
                for preSN in neurons[layer_num]:
                    tmpNeuron.addPreSN(preSN)
        return(neurons)

def defineInputs(listOfVal, neurons):
    """
    Takes in 5 by 5 matrix list of lists of values and defines first 25 inputs of SNN neuron
    """
    count = 0
    for row in listOfVal:
        for value in row:
            neurons[0][count].input = value
            count += 1

def passThrough(neurons):
    for input in neurons[0]:
        isFired = input.doesFire()
        print("({0}, {1:1.3f}, {2:1.3f}) ".format(isFired, input.voltage, input.input), end="")
    print(" / ", end="")
    for layer in range(len(neurons)-1):
        for neuron_num in range(len(neurons[layer + 1])):
            isFired = neurons[layer + 1][neuron_num].doesFire()
           # print("(", isFired, " %.2f)" % neurons[layer+1][neuron_num].voltage, end=" ")
            print("(", isFired, " %.2f)" % neurons[layer + 1][neuron_num].firingAvg, end=" ")
            neurons[layer + 1][neuron_num].updateWeights(10)
            neurons[layer + 1][neuron_num].updateFRAvg(20)
        print(" /", end=" ")
    for neuron in neurons[len(neurons)-1]:
        break
        #print(" " + str(neuron.postSN))
        #print("\n")
    print("\n")


right_input = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]]
no_input = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

nn1 = setupFeedForward(25, 15, 8, 3, 1)
for idx in range(50):
    defineInputs(right_input, nn1)
    for idx2 in range(10):
        passThrough(nn1)
    defineInputs(right_input, nn1)
    for idx3 in range(10):
        passThrough(nn1)

defineInputs(no_input, nn1)
for idx4 in range(50):
    passThrough(nn1)
defineInputs(right_input, nn1)
for idx3 in range(50):
    passThrough(nn1)
input1FR = []
for layer in range(len(nn1)-1):
    input1FR.append([])
    for neuron_num in range(len(nn1[layer + 1])):
        input1FR[layer].append(nn1[layer + 1][neuron_num].firingAvg)

defineInputs(no_input, nn1)
for idx4 in range(50):
    passThrough(nn1)
defineInputs(right_input, nn1)
for idx3 in range(50):
    passThrough(nn1)
input2FR = []
for layer in range(len(nn1) - 1):
    input2FR.append([])
    for neuron_num in range(len(nn1[layer + 1])):
        input2FR[layer].append(nn1[layer + 1][neuron_num].firingAvg)

print(input1FR)
print(input2FR)
