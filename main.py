from network import Network

IMAGE_SIZE= 784
NUM_NEURONS_HIDDEN_LAYER = 128
OUTPUT_LEN = 10

network = Network([IMAGE_SIZE, NUM_NEURONS_HIDDEN_LAYER, NUM_NEURONS_HIDDEN_LAYER, OUTPUT_LEN])

with open("datasets\\mnist_train.csv", "r") as file:
    loss = float("inf")

    for line in file:
        values = line.split(",")

        expected = [0.0] * OUTPUT_LEN
        expected[int(values[0])] = 1.0
        colors = [int(x) / 255 for x in values[1::]]

        network.feed(colors, expected)

        if(network.compute_loss() < loss):
            bestModel = network.save()
        network.backpropagate()

bestModel.save()

accurate = 0
total = 0

with open("datasets\\mnist_test.csv", "r") as file:
    for line in file:
        values = line.split(",")

        number = int(values[0])
        expected = [0.0] * 10
        expected[number] = 1.0
        colors = [int(x) / 255 for x in values[1::]]

        network.feed(colors, expected)

        max = 0.0
        for i, value in enumerate(network.output):
            if value >= max:
                max = value
                predictedNumber = i

        if number == predictedNumber:
            accurate += 1
        
        total += 1

print("Accuracy: ", accurate / total)