from network import Network
import random

network = Network([2, 2, 1])

for i in range(1000000):
    a = random.random()
    b = random.random()
    network.feed([a, b], [a*b])
    network.backpropagate()

print(f"a = {a} | b = {b}")
print(network.output[0])
network.print()