# Spiking-Neural-Network

Requires the following python packages: math, numpy, threading, sys, tkinter

To setup SNN, import neuron.py, this makes use of the following commands:

_net_ = SNN(25, 784, 250, 75, 35, 10) #To initialize the network, 25 refers to number of cycles per input, all numbers after determine the shape of the network
_net_.set_attributes() #Accepts the following optional arguments or none at all to change network attribute values

o	delta_time – sets the simulation time between each run of the network, defaults to 0.05
o	avg_window – sets the number of run-throughs over which the average firing rate of the neurons in the network is calculated
o	learning_rate – dictates the coefficient that multiplies the change to each weight when training via STDP
o	stdp_tau – sets the time constant used in STDP training
o	stdp_offset – sets the STDP offset value which dictates tendency to increase or decrease the weight value
o	max_weight – sets the maximum allowable weight
o	min_weight – sets the minimum allowable weight
o	voltage_threshold – sets the voltage value that must be surpassed for a neuron to fire
o	tau_S – sets the time constant associated with the sum of parent neuron firings
o	tau_R – sets the time constant associated with self neuron firing suppression
o	tau_V – sets the time constant associated with the decay of the voltage
o	tau_threshold – sets the time constant associated with the decay of the voltage threshold in the absence of self neuron firings
o	bias – sets the bias of each neuron
o	delay – sets the time for a pre neuron spike to reach a post spike in terms of number of run-throughs

_net_.neuronPopulate() # initializes the neurons in the SNN
_net_.setupFF() # setup the network as a feedforward network
_net_.setupMNIST(dir1, dir2) # sets up  the network to accpet input from the MNIST dataset. dir1 and dir2 reference file location of MNIST images and MNIST labels
_net_.loadWeights() # loads a weight configuration
_net_.saveWeights() # saves a weight configuration
_net_.deleteSaves() # delete saved weight configurations

# Running the Network
_net_.runSNN(number_of_inputs_to_run_on, list_of_layers_to_train) # first argument dictates how many inputs will be run through network, second argument dictates how many layers will be trained as list that can contain values from 1 to n-1 (where n is number of layers in network)

# Using GUI
The GUI is run from the gui.py module. It will run every setup function that exists in the neuron.py module.
