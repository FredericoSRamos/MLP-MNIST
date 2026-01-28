import random
import math

class Neuron():
    def __init__(self, layer, previous_layer_amount):
        self.bias = random.uniform(-1, 1)
        self.weights = [random.uniform(-1, 1) for _ in range(previous_layer_amount)]
        self.layer = layer
        self.activation = 0
        self.delta = 0

    def activate(self, input):
        sum = self.bias

        for i in range(len(input)):
            sum += input[i].activation * self.weights[i]

        self.activation = 1 / (1 + math.pow(math.e, -sum))

    def update_weight(self, lr, inputs):
        for i in range(len(self.weights)):
            self.weights[i] += lr * self.delta * inputs[i]
            self.bias += lr * self.delta

    def print(self):
        return f"Ativação: {self.activation}. Peso: {self.weights}"