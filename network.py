from neuron import Neuron

class Network():
    def __init__(self, num_neurons):
        if len(num_neurons) < 2:
            raise ValueError("Number of layers must be at least 2")
        
        self.loss = 0
        self.layers = []
        
        previous_num = 0
        for i, num in enumerate(num_neurons):
            layer = [Neuron(i, previous_num) for _ in range(num)]
            self.layers.append(layer)
            previous_num = num

    def feed(self, input, expected):
        if len(expected) != len(self.layers[-1]):
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

    def backpropagate(self, lr=0.05):
        output_layer = self.layers[-1]
        for j, neuron in enumerate(output_layer):
            error = self.expected[j] - neuron.activation
            neuron.delta = error * (neuron.activation * (1 - neuron.activation))

        for i in range(len(self.layers) - 2, 0, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            for j, neuron in enumerate(current_layer):
                error = 0
                for next_neuron in next_layer:
                    error += next_neuron.weights[j] * next_neuron.delta
                neuron.delta = error * (neuron.activation * (1 - neuron.activation))

        for i in range(1, len(self.layers)):
            inputs = [n.activation for n in self.layers[i-1]]
            for neuron in self.layers[i]:
                neuron.update_weight(lr, inputs)
        
        loss = self.compute_loss()
        if self.loss > loss:
            print("Loss = " + str(loss))
            self.loss = loss

    def compute_loss(self):
        sum = 0
        total = 0
        for i, value in enumerate(self.expected):
            sum += pow(value - self.output[i], 2)
            total += 1

        return sum / total

    def print(self):
        for layer in self.layers:
            for neuron in layer:
                print(neuron.print(), end=" ")
            print()