import numpy as np
import math
import random
import pickle


class Input:

    def __init__(self):
        self.doesFirePost = 0
        self.children = []
        self.children_weights = []

        self.sSum = []
        self.tau_1 = 1

    def getSSum(self, idx):
        """
        Updates SSum for parent node and returns value
        """
        self.sSum[idx] = math.exp(-1.0/self.tau_1) * self.sSum[idx] + self.doesFirePost
        return self.sSum[idx]

    def addToChildCurrent(self):
        """
        Adds product of S and weight to each child
        """
        for idx, child in enumerate(self.children):
            child.current += self.children_weights[idx] * self.getSSum(idx)

class Neuron(Input):

    timer = 0
    # subtract 10000 at 5000
    delta_time = 0.05
    avg_window = 100

    def __init__(self):
        super().__init__()
        self.current = 0
        self.voltage = 0
        self.resistance = 5
        self.voltage_threshold = 1
        self.bias = 0
        self.delay = 10   # no of updates for impulse to reach next neuron

        self.doesFire = 0
        self.postFireTimes = []

        self.children_weights = []
        self.child_weight_train = []
        self.self_weight = 1   # Should be between 0 and 1, dictates percent of current that is subtracted from current for suppression

        self.revSSum = 0
        self.tau_2 = 1
        self.tau_0 = 1

        self.current_neuron_time = 10000
        self.children_neuron_times = []

        self.firingRecord = []
        self.firingAvg = 0

        self.neuron_gui = None
        self.child_weight_gui = []

    def addChildConnection(self, neuron, weightValue=None):
        """
        Takes in neuron class and adds it to children, also defines weight of connection, random if no value given
        Also appends zeros to all other required lists
        """
        self.children.append(neuron)
        self.sSum.append(0)
        if weightValue:
            self.children_weights.append(weightValue)
        else:
            self.children_weights.append(np.random.normal(0.75, 1))
        self.child_weight_train.append(0)
        self.children_neuron_times.append(10000)

    def getRevSSum(self):
        """
        Update S value for self-firings to be used to inhibit firings, returns value
        """
        self.revSSum = -math.exp(-1.0/self.tau_2) * self.revSSum - self.doesFire
        return self.revSSum


    def addToSelf(self):
        """
        Adds Reverse S and self weight product and bias to self current
        """
        self.current += self.bias
        self.current += self.self_weight * self.getRevSSum() * self.current

    def updateFiring(self):
        """
        Updates voltage, resetting if it crosses threshold, then updates variable doesFire if neuron fires
        """
        self.voltage += (-self.voltage + self.resistance * self.current) * Neuron.delta_time / float(self.tau_0)
        self.current = 0
        if self.voltage >= self.voltage_threshold:
            self.voltage = 0
            time = self.timer + self.delay
            if time > 5000:
                time -= 10000
            self.postFireTimes.append(time + self.delay)
            self.doesFire = 1
        else:
            self.doesFire = 0
        if self.postFireTimes:
            if self.timer == self.postFireTimes[0]:
                del self.postFireTimes[0]
                self.doesFirePost = 1
            else:
                self.doesFirePost = 0

    def updateAvgFR(self):
        if len(self.firingRecord) < Neuron.avg_window:
            if self.doesFire == 1:
                self.firingRecord.append(1)
            else:
                self.firingRecord.append(0)
            if len(self.firingRecord) == Neuron.avg_window:
                self.firingAvg = 0
                for item in self.firingRecord:
                    self.firingAvg += item
                self.firingAvg = self.firingAvg / float(Neuron.avg_window)
        else:
            remove = self.firingRecord.pop(0)
            add = self.doesFire
            self.firingRecord.append(add)
            self.firingAvg = self.firingAvg + (add - remove)/float(Neuron.avg_window)
        print(self.firingRecord, self.firingAvg)


class SNN:
    def __init__(self, *layers):
        self.input = []
        self.neurons = [[]]
        if len(layers) < 2:
            print(layers)
            print("Not enough layers")
        else:
            for idx in range(layers[0]):
                tmpInput = Input()
                self.input.append(tmpInput)
                tmpInput.children_weights.append(1)
                tmpInput.sSum.append(0)
                self.neurons[0].append(Neuron())
            for layer_num in range(len(layers) - 1):
                self.neurons.append([])
                for idxInLayer in range(layers[layer_num + 1]):
                    self.neurons[layer_num + 1].append(Neuron())

    def __str__(self):
        str = ""
        for neuron_layer in self.neurons:
            for neuron in neuron_layer:
                str += "( {}, {:.1f} ) ".format(neuron.doesFire, neuron.firingAvg)
                # str += "( {}, {:.1f} ) ".format(neuron.doesFire, float(neuron.voltage)/neuron.voltage_threshold)
            str += " / "
        return str

    def setupFF(self):
        for idx, input in enumerate(self.input):
            input.children.append(self.neurons[0][idx])
        for layer_idx, layer in enumerate(self.neurons[:-1]):
            for neuron in layer:
                for child_neuron in self.neurons[layer_idx + 1]:
                    neuron.addChildConnection(child_neuron)

    def runThrough(self):
        for input_n in self.input:
            input_n.addToChildCurrent()
        for neuron_layer in self.neurons[:-1]:
            for neuron in neuron_layer:
                neuron.addToChildCurrent()
                neuron.addToSelf()
                neuron.updateFiring()
                neuron.updateAvgFR()
        for neuron in self.neurons[-1]:
            neuron.addToSelf()
            neuron.updateFiring()
            neuron.updateAvgFR()
        Neuron.timer += 1

    def setInput(self, values):
        """
        Accepts list of values between 0 and 1 for each input neuron
        """
        if len(values) != len(self.neurons[0]):
            print("Wrong number of values")
            return
        for idx, input_neuron in enumerate(self.input):
            input_neuron.doesFirePost = values[idx]

    def saveWeights(self):
        tmp = []
        for layer in range(len(self.neurons)):
            tmp.append([])
            for neuron in range(len(self.neurons[layer])):
                tmp[layer].append([])
                for weight_value in self.neurons[layer][neuron].children_weights:
                    tmp[layer][neuron].append(weight_value)
        print(tmp)
        weight_dict = pickle.load(open("save.p", "rb"))
        print(weight_dict)
        name = input("Enter name for save file: ")
        if name in weight_dict:
            overwrite_check = input("Overwrite existing save? Y/N")
            while overwrite_check.lower() != "y" and overwrite_check.lower() != "n":
                overwrite_check = input("Invalid input")
            if overwrite_check.lower() == "y":
                weight_dict[name.lower()] = tmp
            else:
                pass
        else:
            weight_dict[name.lower()] = tmp
        pickle.dump(weight_dict, open("save.p", "wb"))

    def checkForConsistent(self, weights):  # Checks to make sure that weights is consistent with NN setup accepts weights as input
        tmp_existing_nn = []
        for layer in self.neurons:
            tmp_existing_layer = []
            for neuron in layer:
                tmp_existing_neuron_children = len(neuron.children_weights)
                tmp_existing_layer.append(tmp_existing_neuron_children)
            tmp_existing_nn.append(tmp_existing_layer)

        tmp_request_nn = []
        for layer in weights:
            tmp_existing_layer = []
            for neuron in layer:
                tmp_existing_neuron_children = len(neuron)
                tmp_existing_layer.append(tmp_existing_neuron_children)
            tmp_request_nn.append(tmp_existing_layer)

        return tmp_existing_nn == tmp_request_nn

    def overwriteWeights(self, weights):
        for layer_num in range(len(weights)):
            for neuron_num in range(len(weights[layer_num])):
                print("\nold", self.neurons[layer_num][neuron_num].children_weights)
                self.neurons[layer_num][neuron_num].children_weights = weights[layer_num][neuron_num]
                print("New", self.neurons[layer_num][neuron_num].children_weights)


    def loadWeights(self):
        weight_dict = pickle.load(open("save.p", "rb"))
        for key in weight_dict:
            print(key)
        select = input("Enter a name or 'q' to cancel")
        if select.lower() not in weight_dict:
            while select.lower() not in weight_dict:
                if select.lower() == "q":
                    break
                select = input("That is not a valid save. Enter again")
        if select.lower() in weight_dict:
            new_weights = weight_dict[select.lower()]
            if self.checkForConsistent(new_weights):
                self.overwriteWeights(new_weights)
            else:
                print("Format does not match")





randnum = []
for rand in range(10):
    randnum.append(random.random())


nn1 = SNN(10, 8, 4, 5)
nn1.setupFF()
nn1.setInput(randnum)

#nn1.saveWeights()
nn1.loadWeights()
if __name__ == "__main__":
    for _ in range(100):
        #print(nn1)
        #nn1.runThrough()
        pass