from neuron import Neuron

class Network():
    layers = []
    output = []
    expected = []

    def __init__(self, num_neurons):
        if len(num_neurons) < 2:
            raise ValueError("Number of layers must be at least 2")
        
        previous_num = 0
        for i, num in enumerate(num_neurons):
            layer = [Neuron(i, previous_num) for _ in range(num)]
            self.layers.append(layer)
            previous_num = num

    def feed(self, input, expected):
        if len(expected != len(self.layers[-1])):
            raise ValueError("Size of expected values list cannot be different than number of neuron in last layer")

        if len(input) != len(self.layers[0]):
            raise ValueError("Number of inputs cannot be different than number of neuron in first layer")
        
        self.expected = expected
        
        for i, neuron in enumerate(self.layers[0]):
            neuron.activation = input[i]

        for i in range(1, len(self.layers)):
            for neuron in self.layers[i]:
                neuron.activate(self.layers[i - 1])

        self.output = [neuron.activation for neuron in self.layers[-1]]

    def print(self):
        for layer in self.layers:
            for neuron in layer:
                print(neuron.print(), end=" ")
            print()