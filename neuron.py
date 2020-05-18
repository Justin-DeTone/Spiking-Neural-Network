import numpy as np
import math
import random
import pickle
import read_file
import plot

class Input:
    def __init__(self):
        self.doesFirePost = 0
        self.children = []
        self.children_weights = {}
        self.lastFireTime = None

        self.sSum = {}
        self.tau_1 = 5   # decay of S value

    def getSSum(self, neuron):
        """
        Updates SSum for parent node and returns value
        """
        self.sSum[neuron] = math.exp(-1.0/self.tau_1) * self.sSum[neuron] + self.doesFirePost
        return self.sSum[neuron]

    def addToChildCurrent(self):
        """
        Adds product of S and weight to each child
        """
        for child in self.children:
            child.current += self.children_weights[child] * self.getSSum(child)

    def addToFirstChild(self):
        for child in self.children:
            child.current += self.doesFirePost

class Neuron(Input):

    timer = 0
    # subtract 10000 at 5000
    delta_time = 0.05
    avg_window = 25

    learning_rate = 1
    stdp_time_constant = 1
    stdp_offset = 0.5
    max_weight = 1
    min_weight = -1

    def __init__(self):
        super().__init__()
        self.parents = []
        self.current = 0
        self.voltage = 0
        self.resistance = 1
        self.voltage_threshold = 1
        self.voltage_threshold_attenuation = 1
        self.tau_3 = 1   # time constant for decay of voltage threshold when no firings
        self.bias = 0
        self.delay = 0   # no of updates for impulse to reach next neuron

        self.doesFire = 0
        self.postFireTimes = []

        # self.children_weights = []
        self.child_weight_train = []
        self.self_weight = 1   # Should be between 0 and 1, dictates percent of current that is subtracted from current for suppression

        self.revSSum = 0
        self.tau_2 = 1   # decay of r value
        self.tau_0 = 1  # time constant for decay of voltage value

        self.current_neuron_time = 10000
        self.children_neuron_times = []

        self.firingRecord = []
        self.firingAvg = 0

        self.neuron_gui = None
        self.child_weight_gui = []

    def set_delta_time(self, new_value):
        Neuron.delta_time = new_value

    def set_avg_window(self, new_value):
        Neuron.avg_window = new_value

    def set_learning_rate(self, new_value):
        Neuron.learning_rate = new_value

    def set_stdp_time_constant(self, new_value):
        Neuron.stdp_time_constant = new_value

    def set_stdp_offset(self, new_value):
        Neuron.stdp_offset = new_value

    def set_max_weight(self, new_value):
        Neuron.max_weight = new_value

    def set_min_weight(self, new_value):
        Neuron.min_weight = new_value

    def set_voltage_threshold(self, new_value):
        self.voltage_threshold = new_value

    def set_tau0(self, new_value):
        self.tau_0 = new_value

    def set_tau1(self, new_value):
        self.tau_1 = new_value

    def set_tau2(self, new_value):
        self.tau_2 = new_value

    def set_tau3(self, new_value):
        self.tau_3 = new_value

    def set_bias(self, new_value):
        self.bias = new_value

    def set_delay(self, new_value):
        self.delay = new_value




    def addChildConnection(self, child, weightValue=None):
        self.children.append(child)
        self.sSum[child] = 0
        if weightValue:
            self.children_weights[child] = weightValue
        else:
            weight = self.__getRandomWeightValue(0.125, 0.25)
            self.children_weights[child] = weight
        self.child_weight_train.append(0)
        self.children_neuron_times.append(10000)

    def __getRandomWeightValue(self, mean, dev):
        weight = np.random.normal(mean, dev)
        if weight > Neuron.max_weight:
            return Neuron.max_weight
        elif weight < Neuron.min_weight:
            return Neuron.min_weight
        else:
            return weight

    def addParentConnection(self, neuron):
        self.parents.append(neuron)

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

    def updateVoltage(self):
        self.voltage += (-self.voltage + self.resistance * self.current) * Neuron.delta_time / float(self.tau_0)
        self.current = 0

    def simpleUpdateVoltage(self):
        self.voltage += self.voltage_threshold / 5 * self.current
        self.current = 0

    def updateFiring(self):
        """
        updates variable doesFire if neuron fires
        """

        if self.voltage >= self.voltage_threshold * self.voltage_threshold_attenuation:
            self.voltage = 0
            time = self.timer + self.delay
            if time > 5000:
                time -= 10000
            self.postFireTimes.append(time)
            self.doesFire = 1
            self.lastFireTime = self.timer
        else:
            self.doesFire = 0
        if self.postFireTimes:
            if self.timer == self.postFireTimes[0]:
                del self.postFireTimes[0]
                self.doesFirePost = 1
            else:
                self.doesFirePost = 0
        else:
            self.doesFirePost = 0

    def adjustAtten(self):
        if self.lastFireTime:
            time_delta = (self.timer - self.lastFireTime) * Neuron.delta_time
        else:
            time_delta = self.timer * Neuron.delta_time
        self.voltage_threshold_attenuation = math.exp(-time_delta/self.tau_3)


    def resetLastFire(self):
        if self.lastFireTime:
            self.lastFireTime -= 10000
            if self.lastFireTime < -5000:
                self.lastFireTime = None

    def updateAvgFR(self):
        self.firingRecord.append(self.doesFire)
        divisor = len(self.firingRecord)
        if len(self.firingRecord) >= Neuron.avg_window:
            self.firingRecord.pop(0)
            divisor = self.avg_window
        result = 0
        for doesFire in self.firingRecord:
            result += doesFire
        self.firingAvg = result / divisor

    def stdp(self):
        if self.doesFire == 1:
            for parent_neuron in self.parents:
                pre_weight = parent_neuron.children_weights[self]
                weight_value = parent_neuron.children_weights[self]
                time_pre = parent_neuron.lastFireTime
                if not time_pre:
                    continue
                time_post = self.lastFireTime
                time_delta = (time_pre-time_post) * self.delta_time
                time_comp = math.exp(time_delta / self.stdp_time_constant) - self.stdp_offset
                weight_delta = self.learning_rate * time_comp * (self.max_weight - weight_value) * \
                    (weight_value - self.min_weight)
                parent_neuron.children_weights[self] = parent_neuron.children_weights[self] + weight_delta
                post_weight = parent_neuron.children_weights[self]
                print("Pre: {:.3f}; Post: {:.3f}; Diff: {:.3f}".format(pre_weight, post_weight, weight_delta))

        # for parent_neuron in self.parents:
        #         #     print("current weight: {}".format(parent_neuron.children_weights[self]))

class SNN:
    def __init__(self, max_count, *layers):
        self.count = 0   # count to change input
        self.max_count = max_count
        self.image_idx = 0
        self.training_data = None
        self.test_data = None
        self.currentNumber = None
        self.dictOfAttributeAdjustments = {}
        self.count_to_five = 0

        self.middle_layers_idx = []

        self.input = []
        self.neurons = [[]]
        self.layers = layers

        self.image_dir = None
        self.label_dir = None

        self.plot_reference = plot.Plot()

    def __str__(self):
        snn_string = ""
        for neuron_layer in self.neurons:
            for neuron in neuron_layer:
                snn_string += "( {}, {:.1f} ) ".format(neuron.doesFire, neuron.firingAvg)
                # str += "( {}, {:.1f} ) ".format(neuron.doesFire, float(neuron.voltage)/neuron.voltage_threshold)
            snn_string += " / "
        return snn_string

    def set_attributes(self, delta_time=None, avg_window=None, learning_rate=None, stdp_tau=None, stdp_offset=None,
                       max_weight=None, min_weight=None, voltage_threshold=None, tau_S=None, tau_R=None,
                       tau_V=None, tau_Threshold=None, bias=None, delay=None):
        if delta_time:
            self.dictOfAttributeAdjustments["delta_time"] = delta_time
        if avg_window:
            self.dictOfAttributeAdjustments["avg_window"] = avg_window
        if learning_rate:
            self.dictOfAttributeAdjustments["learning_rate"] = learning_rate
        if stdp_tau:
            self.dictOfAttributeAdjustments["stdp_tau"] = stdp_tau
        if stdp_offset:
            self.dictOfAttributeAdjustments["stdp_offset"] = stdp_offset
        if max_weight:
            self.dictOfAttributeAdjustments["max_weight"] = max_weight
        if min_weight:
            self.dictOfAttributeAdjustments["min_weight"] = min_weight
        if voltage_threshold:
            self.dictOfAttributeAdjustments["voltage_threshold"] = voltage_threshold
        if tau_S:
            self.dictOfAttributeAdjustments["tau_S"] = tau_S
        if tau_R:
            self.dictOfAttributeAdjustments["tau_R"] = tau_R
        if tau_V:
            self.dictOfAttributeAdjustments["tau_V"] = tau_V
        if tau_Threshold:
            self.dictOfAttributeAdjustments["tau_Threshold"] = tau_Threshold
        if bias:
            self.dictOfAttributeAdjustments["bias"] = bias
        if delay:
            self.dictOfAttributeAdjustments["delay"] = delay

    def updateNeuronAttributesClass(self, neuron):
        if "delta_time" in self.dictOfAttributeAdjustments:
            neuron.set_delta_time(self.dictOfAttributeAdjustments["delta_time"])
        if "avg_window" in self.dictOfAttributeAdjustments:
            neuron.set_avg_window(self.dictOfAttributeAdjustments["avg_window"])
        if "learning_rate" in self.dictOfAttributeAdjustments:
            neuron.set_learning_rate(self.dictOfAttributeAdjustments["learning_rate"])
        if "stdp_tau" in self.dictOfAttributeAdjustments:
            neuron.set_stdp_time_constant(self.dictOfAttributeAdjustments["stdp_tau"])
        if "stdp_offset" in self.dictOfAttributeAdjustments:
            neuron.set_stdp_offset(self.dictOfAttributeAdjustments["stdp_offset"])
        if "max_weight" in self.dictOfAttributeAdjustments:
            neuron.set_max_weight(self.dictOfAttributeAdjustments["max_weight"])
        if "min_weight" in self.dictOfAttributeAdjustments:
            neuron.set_min_weight(self.dictOfAttributeAdjustments["min_weight"])

    def updateNeuronAttributesInstance(self, neuron):
        if "voltage_threshold" in self.dictOfAttributeAdjustments:
            neuron.set_voltage_threshold(self.dictOfAttributeAdjustments["voltage_threshold"])
        if "tau_V" in self.dictOfAttributeAdjustments:
            neuron.set_tau0(self.dictOfAttributeAdjustments["tau_V"])
        if "tau_S" in self.dictOfAttributeAdjustments:
            neuron.set_tau1(self.dictOfAttributeAdjustments["tau_S"])
        if "tau_R" in self.dictOfAttributeAdjustments:
            neuron.set_tau2(self.dictOfAttributeAdjustments["tau_R"])
        if "tau_Threshold" in self.dictOfAttributeAdjustments:
            neuron.set_tau3(self.dictOfAttributeAdjustments["tau_Threshold"])
        if "bias" in self.dictOfAttributeAdjustments:
            neuron.set_bias(self.dictOfAttributeAdjustments["bias"])
        if "delay" in self.dictOfAttributeAdjustments:
            neuron.set_delay(self.dictOfAttributeAdjustments["delay"])

    def __getLastLayerIndex(self): # call after 1
        self.last_layer_idx = [len(self.layers) - 3]

    def __getMiddleLayerIndex(self):  # call after init 2
        for number in range(len(self.layers) - 2):
            self.middle_layers_idx.append(number + 1)

    def neuronPopulate(self):
        if len(self.layers) < 2:
            print(self.layers)
            print("Not enough layers, need minimum of 2")
        else:
            self.__initNeurons()
            self.__getLastLayerIndex()
            self.__getMiddleLayerIndex()

    def __initNeurons(self):
        for idx in range(self.layers[0]):
            tmpInput = Input()
            self.input.append(tmpInput)
            neuron = Neuron()
            self.neurons[0].append(neuron)
            tmpInput.children_weights[neuron] = 1
            tmpInput.sSum[neuron] = 0
        for layer_num in range(len(self.layers) - 1):
            self.neurons.append([])
            for idxInLayer in range(self.layers[layer_num + 1]):
                newNeuron = Neuron()
                self.updateNeuronAttributesInstance(newNeuron)
                self.neurons[layer_num + 1].append(newNeuron)
        self.updateNeuronAttributesClass(self.neurons[0][0])



    def setupFF(self):
        for idx, input in enumerate(self.input):
            input.children.append(self.neurons[0][idx])
        for layer_idx, layer in enumerate(self.neurons[:-1]):
            for neuron in layer:
                for child_neuron in self.neurons[layer_idx + 1]:
                    neuron.addChildConnection(child_neuron)
                if layer_idx > 0:
                    for parent_neuron in self.neurons[layer_idx - 1]:
                        neuron.addParentConnection(parent_neuron)

    def runOnLayer(self, current_layer, function, run_layer_list):
        if current_layer in run_layer_list:
            function()

    def runThrough(self, stdp_active_layers_list=None):
        if stdp_active_layers_list is None:
            stdp_active_layers_list = []

        for input_n in self.input:
            input_n.addToFirstChild()
        for idx, neuron_layer in enumerate(self.neurons[:-1]):
            for neuron in neuron_layer:
                # Last layer does not run this
                self.runOnLayer(idx, neuron.addToChildCurrent, [0] + self.middle_layers_idx)
                # All neurons run these
                neuron.addToSelf()
                # First Layer runs different updateVoltage
                self.runOnLayer(idx, neuron.simpleUpdateVoltage, [0])
                self.runOnLayer(idx, neuron.updateVoltage, self.middle_layers_idx + self.last_layer_idx)
                # All neurons run these
                neuron.updateFiring()
                neuron.updateAvgFR()
                #Selective neurons run this
                self.runOnLayer(idx, neuron.stdp, stdp_active_layers_list)
                #All neurons run these
                neuron.adjustAtten()

        Neuron.timer += 1
        if Neuron.timer > 5000:
            Neuron.timer -= 10000
            for neuron_layer in self.neurons:
                for neuron in neuron_layer:
                    neuron.resetLastFire()
        self.count += 1
        self.count_to_five += 1

    def setNormalizedInput(self, values):
        """
        Accepts list of values between 0 and 1 for each input neuron
        """
        if len(values) != len(self.neurons[0]):
            print("Wrong number of values")
            return
        for idx, input_neuron in enumerate(self.input):
            input_neuron.doesFirePost = values[idx]

    def deleteSaves(self):
        weight_dict = pickle.load(open("save.p", "rb"))
        print("Existing files:")
        for name in weight_dict:
            print(name, ": ", weight_dict[name])
        key_del = input("Enter filename to delete: ")
        if key_del in weight_dict:
            check = input("Are you sure you want to delete save: {}? (y/n)".format(key_del))
            if check.lower() == "y":
                del weight_dict[key_del]
                print(key_del, " has been deleted")
        pickle.dump(weight_dict, open("save.p", "wb"))
        print("Existing files:")
        for name in weight_dict:
            print(name, ": ", weight_dict[name])


    def saveWeights(self):
        key = self.generateNeuronKey()
        tmp = []
        for layer in range(len(self.neurons)):
            tmp.append([])
            for neuron in range(len(self.neurons[layer])):
                tmp[layer].append({})
                for child in self.neurons[layer][neuron].children_weights:
                    tmp[layer][neuron][key[layer][child]] = self.neurons[layer][neuron].children_weights[child]
        weight_dict = pickle.load(open("save.p", "rb"))
        print("Existing files:")
        for name in weight_dict:
            print(name, ": ", weight_dict[name])
        name = input("Enter name to save file under: ")

        tmp_list = []
        for key in weight_dict:
            tmp_list.append(key.lower())

        while name.lower() in tmp_list:
            overwrite_check = input("Overwrite existing save? Y/N")
            while overwrite_check.lower() != "y" and overwrite_check.lower() != "n":
                overwrite_check = input("Invalid input")
            if overwrite_check.lower() == "y":
                break
            else:
                name = input("Enter a different name for save file: ")

        weight_dict[name] = tmp
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

    def generateIDKey(self):
        neuronKey = []
        for idx, layer in enumerate(self.neurons):
            neuronKey.append({})
            for neuron in layer:
                counter = 1
                for child in neuron.children_weights:
                    if child not in neuronKey[idx].values():
                        while counter in neuronKey[idx]:
                            counter += 1
                        neuronKey[idx][counter] = child
        return neuronKey

    def generateNeuronKey(self):
        neuronKey = []
        for idx, layer in enumerate(self.neurons):
            neuronKey.append({})
            for neuron in layer:
                counter = 1
                for child in neuron.children_weights:
                    if child not in neuronKey[idx].keys():
                        while counter in neuronKey[idx].values():
                            counter += 1
                        neuronKey[idx][child] = counter
        return neuronKey

    def overwriteWeights(self, weights):
        key = self.generateIDKey()
        for layer_num in range(len(weights)):
            for neuron_num in range(len(weights[layer_num])):
                print("\nold", self.neurons[layer_num][neuron_num].children_weights)
                self.neurons[layer_num][neuron_num].children_weights = {}
                for child in weights[layer_num][neuron_num]:
                    # print(child)
                    # print(key)
                    # print(key[child])
                    self.neurons[layer_num][neuron_num].children_weights[key[layer_num][child]] = weights[layer_num][neuron_num][child]
                print("New", self.neurons[layer_num][neuron_num].children_weights)

    def loadWeights(self):
        weight_dict = pickle.load(open("save.p", "rb"))
        for key in weight_dict:
            print(key)
        select = input("Enter a name from above to load from or enter 'q' to cancel")
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

    def convertInput(self, list2):
        pixel_list = []
        for pixel in list2[1]:
            pixel_list.append(pixel/255)
        self.setNormalizedInput(pixel_list)
        self.currentNumber = list2[0][0]

    def checkNewInput(self):
        if self.count == 1:
            print(self.image_idx)
        if self.count >= self.max_count:
            self.count = 0
            self.image_idx += 1
            self.convertInput(read_file.return_image(self.image_dir, self.label_dir, self.image_idx))

    def returnLastWeightValues(self):
        key_N2num = self.generateNeuronKey()
        output = ""
        for parent in self.neurons[-2]:
            output += "{} | ".format(key_N2num[-3][parent])
            for child in parent.children_weights:
                weight = parent.children_weights[child]
                num = key_N2num[-2][child]
                output += "{}: {:.3f}, ".format(num, weight)
            output += "\n"
        return output

    def setupMNIST(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        img_list = read_file.return_image(image_dir, label_dir, self.image_idx)
        self.convertInput(img_list)

    def runSNN(self, no_of_images, layers_to_train=None):
        if layers_to_train is None:
            layers_to_train = []
        for layer in layers_to_train:
            if layer >= len(self.neurons):
                print("invalid layer entered")
                return
            if layer == 0:
                print("cannot update with 0 as post layer")
                return
        for count in range(self.max_count * no_of_images):
            self.runCalculateGloabalFireRate()
            self.checkNewInput()
            self.runThrough([1])
        self.plot_reference.plotPlot()

    def runCalculateGloabalFireRate(self):
        if self.count_to_five % 5 == 0:
            self.plot_reference.addPoint(self.count_to_five, self.calculateGlobalFireRate())

    def calculateGlobalFireRate(self):
        divisor = 0
        numerator = 0
        for layer in self.neurons[1:]:
            for neuron in layer:
                divisor += 1
                numerator += neuron.firingAvg
        return numerator / divisor




randnum = []
for rand in range(10):
    randnum.append(random.random())
# nn_test = SNN(200, 20, 10, 5, 1)
# nn_test.setupFF()

nn1 = SNN(25, 784, 250, 75, 35, 10)
nn1.set_attributes(stdp_offset=0.368, stdp_tau=0.5, tau_S=100, tau_R=100, voltage_threshold=25, tau_Threshold=100)
nn1.neuronPopulate()
nn1.setupFF()

# img_list = read_file.return_image('./mnist/train-images.idx3-ubyte', './mnist/train-labels.idx1-ubyte', nn1.image_idx)
# # nn1.convertInput(img_list)
nn1.setupMNIST('./mnist/train-images.idx3-ubyte', './mnist/train-labels.idx1-ubyte')
# nn1.loadWeights()
#nn1.saveWeights()
#nn1.deleteSaves()
#nn1.loadWeights()


if __name__ == "__main__":
    nn1.runSNN(10, [1])
