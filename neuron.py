import random

class Neuron():
    activation = 0
    weights = []

    def __init__(self, layer, previous_layer_amount):
        if layer != 0:
            self.bias = random.random()
            self.weights = [random.random() for _ in range(previous_layer_amount)]
        self.layer = layer

    def activate(self, input):
        sum = 0

        for i in input:
            for j in self.weights:
                sum += i.activation * j

        self.activation = max(0, sum)

    def print(self):
        return f"Ativação: {self.activation}. Peso: {self.weights}"