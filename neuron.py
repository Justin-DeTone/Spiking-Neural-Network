import numpy as np
import math
import random
import pickle
import queue
import read_file

class Input:
    def __init__(self):
        self.doesFirePost = 0
        self.children = []
        self.children_weights = {}
        self.lastFireTime = None

        self.sSum = {}
        self.tau_1 = 5

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
    stdp_offset = 0.7
    max_weight = 1
    min_weight = -1

    def __init__(self):
        super().__init__()
        self.parents = []
        self.current = 0
        self.voltage = 0
        self.resistance = 1
        self.voltage_threshold = 5
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
        self.tau_2 = 5
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
        self.sSum[neuron] = 0
        if weightValue:
            self.children_weights[neuron] = weightValue
        else:
            weight = np.random.normal(0.125, 0.25)
            if weight > Neuron.max_weight:
                self.children_weights[neuron] = Neuron.max_weight
            elif weight < Neuron.min_weight:
                self.children_weights[neuron] = Neuron.min_weight
            else:
                self.children_weights[neuron] = weight
        self.child_weight_train.append(0)
        self.children_neuron_times.append(10000)

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
        print(self.voltage_threshold_attenuation)


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
                # print("Pre: {:.3f}; Post: {:.3f}; Diff: {:.3f}".format(pre_weight, post_weight, weight_delta))

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

        self.input = []
        self.neurons = [[]]
        if len(layers) < 2:
            print(layers)
            print("Not enough layers")
        else:
            for idx in range(layers[0]):
                tmpInput = Input()
                self.input.append(tmpInput)
                neuron = Neuron()
                self.neurons[0].append(neuron)
                tmpInput.children_weights[neuron] = 1
                tmpInput.sSum[neuron] = 0
            for layer_num in range(len(layers) - 1):
                self.neurons.append([])
                for idxInLayer in range(layers[layer_num + 1]):
                    self.neurons[layer_num + 1].append(Neuron())
        self.just_fired_neurons = queue.Queue() # Queue for neurons that just fired and need to update weights MOVE
        self.last_layer_idx = [len(layers) - 3]
        nums = range(len(layers) - 2)
        self.middle_layers_idx = []
        for number in nums:
            self.middle_layers_idx.append(number + 1)

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
                if layer_idx > 0:
                    for parent_neuron in self.neurons[layer_idx - 1]:
                        neuron.addParentConnection(parent_neuron)

    def runOnLayer(self, current_layer, function, run_layer_list):
        if current_layer in run_layer_list:
            function()

    def runThrough(self):
        # if self.count >= self.max_count:
        #     self.count = 0
        #     self.image_idx += 1
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
                self.runOnLayer(idx, neuron.stdp, [1])
                #All neurons run these
                neuron.adjustAtten()
        #     for neuron in neuron_layer:
        #         neuron.addToChildCurrent()
        #         neuron.addToSelf()
        #         if idx == 0:
        #             neuron.simpleUpdateVoltage()
        #         else:
        #             neuron.updateVoltage()
        #         neuron.updateFiring()
        #         neuron.updateAvgFR()
        #         self.runOnLayer(idx, neuron.stdp(), [1])
        #         neuron.adjustAtten()
        # for neuron in self.neurons[-1]:
        #     neuron.addToSelf()
        #     neuron.updateVoltage()
        #     neuron.updateFiring()
        #     neuron.updateAvgFR()
        #     neuron.stdp()
        Neuron.timer += 1
        if Neuron.timer > 5000:
            Neuron.timer -= 10000
            for neuron_layer in self.neurons:
                for neuron in neuron_layer:
                    neuron.resetLastFire()
        self.count += 1

    def setInput(self, values):
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

    def adjustWeights(self):   # Performs STDP unsupervised learning for first neuron in queue of neurons that just fired
        fired_neuron = self.just_fired_neurons.get()   # item in queue is format: [neuron_instance, # in layer]
        for pre_neuron in fired_neuron[0].parents:
            #get position of fired_neuron in layer, that is the idx used to get weight in pre_neuron
            idx_weight = fired_neuron[1]

    def convertInput(self, list2):
        pixel_list = []
        for pixel in list2[1]:
            pixel_list.append(pixel/255)
        self.setInput(pixel_list)
        self.currentNumber = list2[0][0]

    def checkNewInput(self):
        if self.count >= self.max_count:
            self.count = 0
            self.image_idx += 1
            self.convertInput(read_file.return_image('./mnist/train-images.idx3-ubyte', './mnist/train-labels.idx1-ubyte', self.image_idx))

    def returnLastWeightValues(self):
        # key_num2N = self.generateIDKey()
        # output = ""
        # for parent in self.neurons[-2]:
        #     for num in range(len(parent.children_weights)):
        #         neuron = key_num2N[-2].get(num + 1, None)
        #         if not neuron:
        #             continue
        #         output += "{}: {:.3f}, ".format(num+1, parent.children_weights[neuron])
        #     output += "\n"
        # return output
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


randnum = []
for rand in range(10):
    randnum.append(random.random())
# nn_test = SNN(200, 20, 10, 5, 1)
# nn_test.setupFF()

nn1 = SNN(50, 784, 250, 75, 35, 10)
nn1.setupFF()

img_list = read_file.return_image('./mnist/train-images.idx3-ubyte', './mnist/train-labels.idx1-ubyte', nn1.image_idx)
nn1.convertInput(img_list)
weights_before = nn1.returnLastWeightValues()

#nn1.saveWeights()
#nn1.deleteSaves()
#nn1.loadWeights()
if __name__ == "__main__":
    for num in range(5*50):
        #print(_, nn1)
        # print(nn1.image_idx)
        # print(nn1.returnLastWeightValues())
        nn1.checkNewInput()
        nn1.runThrough()
        pass

    # print(weights_before, "\n", nn1.returnLastWeightValues())
