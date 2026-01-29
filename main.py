from network import Network
import random

network = Network([2, 2, 1])

loss = float("inf")
for i in range(1000000):
    a = random.random()
    b = random.random()
    network.feed([a, b], [a*b])
    if(network.compute_loss() < loss):
        bestModel = network.save()
    network.backpropagate()

bestModel.save()

network = Network.loadFromFile("models\\network_1769811662.611141")
network.feed([0.5, 0.9], [0.45])
print(network.output[0])
network.print()