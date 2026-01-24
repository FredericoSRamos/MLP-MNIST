from network import Network

network = Network([2, 2, 1])
network.feed([1, 2])
print(network.output)
network.print()