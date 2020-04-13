import tkinter as tk
import neuron as n
import color_convert as cc
import math
import threading
import sys

"""
root = tkinter.Tk()
w = tkinter.Canvas(root, width=500, height=500)
w.pack()
root.mainloop()
"""

refresh_time = 0.3
is_paused = False

class NeuronDisplay:
    def __init__(self, x_coor, y_coor, radius, neuron, parent):
        self.neuron = neuron
        self.x_coor = x_coor
        self.y_coor = y_coor
        self.radius = radius

        self.children = []
        self.parent = parent
        self.text = None
        self.item = None

    def getColor(self, base, neuron):
        # takes as input of hex color to use as base color and returns hex color based on neuron firing frequency
        rgb = cc.hexToRGB(base)
        rgb2 = []
        for color in rgb:
            rgb2.append(int(255 - (math.pow(neuron.firingAvg, 0.5)) * (255 - color)))
        return cc.rgbToHex(rgb2)

    def getColor2(self, base, neuron):
        # takes as input color to display when neuron fires
        if neuron.doesFire == 1:
            return base
        else:
            return "#ffffff"

    def display(self):
        self.item = w.create_oval(self.x_coor - self.radius, self.y_coor - self.radius, self.x_coor + self.radius,
                                  self.y_coor + self.radius, fill=self.getColor("#ff6600", self.neuron))
        if self.radius > 6:
            self.text = w.create_text(self.x_coor, self.y_coor, text="{:.1f}".format(self.neuron.firingAvg))

    def updateColor(self):
        w.itemconfig(self.item, fill=self.getColor("#ff6600", self.neuron))

    def updateText(self):
        if self.radius > 6:
            w.itemconfig(self.text, text="{:.1f}".format(self.neuron.firingAvg))
        pass

class WeightDisplay:
    def __init__(self, x_coor1, y_coor1, x_coor2, y_coor2, parent):
        self.x1 = x_coor1
        self.y1 = y_coor1
        self.x2 = x_coor2
        self.y2 = y_coor2
        self.parent = parent
        self.item = None

    def display(self):
        self.item = w.create_line(self.x1, self.y1, self.x2, self.y2)

    def update(self):
        w.itemconfig(self.item)

class SNNDisplay:
    def __init__(self, neurons):
        self.neurons = neurons
        self.neuronDisp = []
        self.width_diff = 0
        self.height_diff = []
        self.radius = []

    def getRadius(self):
        for layer in self.neurons:
            self.radius.append(len(layer))

        #find max radius by height:
        for radius in self.radius:
            self.height_diff.append(can_height / radius)
        self.width_diff = can_width / len(self.neurons)

        for idx, layer_num in enumerate(self.radius):
            self.radius[idx] = can_height/layer_num * 2/5

        width_radius = can_width/(6 * len(self.neurons))

        # If width radius is lower over ride that layer:
        for idx, radius in enumerate(self.radius):
            if radius > width_radius:
                self.radius[idx] = width_radius

        # maxNPerLayer = 0
        # for layer in self.neurons:
        #     if len(layer) > maxNPerLayer:
        #         maxNPerLayer = len(layer)
        # self.height_diff = can_height / maxNPerLayer
        #
        # self.width_diff = can_width / len(self.neurons)
        #
        # # Calculate radius according to height spacing
        # self.radius = can_height * 2/(5 * maxNPerLayer)
        #
        # #Calculate radius according to width spacing
        # tmp = can_width/(6 * len(self.neurons))
        #
        # #Choose lower radius
        # # if tmp < self.radius:
        # #     self.radius = tmp

    def initNeurons(self):
        self.getRadius()
        x_coor = self.width_diff / 2
        for idx, layer in enumerate(self.neurons):
            self.neuronDisp.append([])
            y_coor = can_height / 2
            y_coor -= self.height_diff[idx] * (len(layer) - 1)/2
            for neuron in layer:
                tmp = NeuronDisplay(x_coor, y_coor, self.radius[idx], neuron, master)
                tmp.neuron = neuron # set Neuron = neuron
                neuron.neuron_gui = tmp
                self.neuronDisp[-1].append(tmp) # append to SNNDisplay
                y_coor += self.height_diff[idx]
            x_coor += self.width_diff

    def initWeights(self):
        for layer_num in range(len(self.neurons)-1):
            for neuron in self.neurons[layer_num]:
                for child in neuron.children:
                    tmp = WeightDisplay(neuron.neuron_gui.x_coor, neuron.neuron_gui.y_coor,
                                        child.neuron_gui.x_coor, child.neuron_gui.y_coor, master)
                    neuron.child_weight_gui.append(tmp)

    def displaySNN(self):
        for layer in self.neurons:
            for neuron in layer:
                for weight in neuron.child_weight_gui:
                    weight.display()
                neuron.neuron_gui.display()

    def updateSNN(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.neuron_gui.updateColor()
                neuron.neuron_gui.updateText()

"""
class UpdateSNN(threading.Thread):
    def __init__(self, network):
        super().__init__()
        self.network = network
    def __run__(self):
        self.network.runThrough()
        event = threading.Timer(refresh_time, updateSNN)
        
def updateSNN():
    tmp = UpdateSNN(n.nn1)
    tmp.start()
"""

def updateSNN(network):
    if not is_paused:
        network.runThrough()
    print(network)
    next_ev = threading.Timer(refresh_time, updateSNN, [network])
    next_ev.start()

def pause():
    global is_paused
    is_paused = not is_paused

def close():
    sys.exit()

if __name__ == "__main__":
    can_width = 1000
    can_height = 700

    master = tk.Tk()

    f = tk.Frame(master, width=900)
    f.pack_propagate(1)
    f.pack()

    pause = tk.Button(f, text="Pause", command=pause)
    pause.pack()

    w = tk.Canvas(master, width=can_width, height=can_height)
    w.pack()

    a = SNNDisplay(n.nn1.neurons)
    a.initNeurons()
    #a.initWeights()
    a.displaySNN()

    master.protocol("WM_DELETE_WINDOW", close)

    # tk.mainloop()
    updateSNN(n.nn1)

    while 1:
        a.updateSNN()
        master.update_idletasks()
        master.update()





