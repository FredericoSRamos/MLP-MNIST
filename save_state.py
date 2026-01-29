from datetime import datetime
import pickle

class SaveState:
    numNeurons: list[int]
    weights: list[list[list[float]]]
    biases: list[list[float]]

    def __init__(self, numNeurons=None, weights=None, biases=None):
        self.numNeurons = numNeurons
        self.weights = weights
        self.biases = biases

    def save(self, filePath=None):
        if filePath == None:
            filePath = f"models\\network_{datetime.now().timestamp()}"
        with open(filePath, "xb") as file:
            pickle.dump(self, file)

    def load(filePath):
        if filePath is None:
            raise ValueError("A path must be specified")
        
        with open(filePath, "rb") as file:
            return pickle.load(file)
