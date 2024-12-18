import numpy as np
from tqdm import tqdm

from aux import convert_from_file_MOD

print("\n\n\n\n\n\n\n\n\n\n")



def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    sigmoid_value = sigmoid(x)
    return (sigmoid_value * (1 - sigmoid_value))


def generateVec(s):
    #Expect s to be an integer between 0 and 9
    res = np.zeros((10,1))
    res[s] = 1
    return res

class layer():
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        #  Weights:
        #                  rows, columns
        #self.W = np.zeros((n_in, n_out))
        self.W = np.random.randn(n_in, n_out)
        #self.W = sigmoid(np.random.randn((n_in, n_out)))
        self.B = np.zeros((n_out, 1))
        self.Y = np.zeros((n_out, 1))
        self.Z = np.zeros((n_out, 1))
        self.act_func = sigmoid
        self.act_func_der = sigmoid_derivative
        # for backpropagation:
        self.delta = np.zeros((n_out, 1))

    def forward(self, inputNeurons):
        self.Y = self.W.T @ inputNeurons + self.B
        #self.Y = np.dot(self.W.T, inputNeurons) + self.B
        self.Z = self.act_func(self.Y)

    def update(self, alpha, inputVec):
        self.W = self.W - alpha*( inputVec @ self.delta.T )
        self.B = self.B - alpha*self.delta


class NeuralNetwork():
    def __init__(self, listLayers):
        self.listLayers = listLayers
        self.numLayers = len(self.listLayers)

    def getOutput(self, givenInput):
        return self.listLayers[-1].Z

    def forward(self, givIn):
        #print("Start forward propagation.")
        # For the first layer:
        self.listLayers[0].forward(givIn)
        # For the other layers:
        for j in range(1, self.numLayers):
            self.listLayers[j].forward(self.listLayers[j-1].Z)
        #print("Finished forward propagation.")

    def backward(self, givenInput, givTarget, rate):
        #print("Starting back-propagation.")
        # Start by the last layer:
        tempLayer = self.listLayers[-1]
        bigDelta = tempLayer.Z - givTarget
        f_prime_Y = tempLayer.act_func_der(tempLayer.Y)
        #tempLayer.delta = np.array([ bigDelta[l]*f_prime_Y[l] for l in range(tempLayer.n_out)])
        self.listLayers[-1].delta = bigDelta * f_prime_Y
        # Do the other layers:
        for j in range(1, self.numLayers):
            s = self.numLayers -1-j
            tempLayer = self.listLayers[s]
            laterLayer = self.listLayers[s+1]
            f_prime_Y = tempLayer.act_func_der(tempLayer.Y)
            vecV = laterLayer.W @ laterLayer.delta
            #tempLayer.delta = np.array([ f_prime_Y[l]*vecV[l] for l in range(tempLayer.n_out) ])
            #tempLayer.delta = f_prime_Y * vecV
            self.listLayers[s].delta = np.array([ f_prime_Y[l]*vecV[l] for l in range(tempLayer.n_out) ])
        #print("Backpropagation is over. ")
        #print("Updating the weights.")
        for j in range(self.numLayers):
            if j==0:
                useIn = givenInput
            else:
                useIn = self.listLayers[j-1].Z
            self.listLayers[j].update(rate, useIn)
        #print("Finished updating the weights.")

    def train(self, inputList, targetList, learn_rate, numEpochs=1):
        numInputs = len(inputList)
        print("Begin training.")
        print("Number of training iterations:  "+str(numInputs))
        for j in range(numEpochs):
            print("\t Epoch nº "+str(j+1))
            #for k in range(numInputs):
            for k in tqdm(range(numInputs)):
                thisInput = inputList[k]
                thisOutput = targetList[k]
                # 1) forward
                self.forward(thisInput)
                # 2) backpropagation:
                self.backward(thisInput, thisOutput, learn_rate)
        print("Finished training.")

    def test(self, inputTest, targtetTest):
        numInputs = len(inputTest)
        print("Begin testing.")
        print("Number of testing iterations:  "+str(numInputs))
        correctPredictions = 0
        for j in tqdm(range(numInputs)):
            thisInput = inputTest[j]
            thisOutput = targtetTest[j]
            # 1) forward:
            self.forward(thisInput)
            # 2) Get error of prediction:
            thisIterZ = self.getOutput(thisInput)
            if np.argmax(thisIterZ) == thisOutput:
                correctPredictions +=1
        print("Finished testing.")
        accuracy_percentage = (correctPredictions/numInputs)*100
        return accuracy_percentage





# Load MNIST data:
path_training_set = "MNIST_DATABASE/train-images.idx3-ubyte"
training_set = convert_from_file_MOD(path_training_set)

path_training_labels = "MNIST_DATABASE/train-labels.idx1-ubyte"
training_labels = convert_from_file_MOD(path_training_labels)

path_testing_set = "MNIST_DATABASE/t10k-images.idx3-ubyte"
testing_set = convert_from_file_MOD(path_training_set)

path_testing_labels = "MNIST_DATABASE/t10k-labels.idx1-ubyte"
testing_labels = convert_from_file_MOD(path_training_labels)



# Define the network
hiddenLayer = layer(784, 64)
outputLayer = layer(64, 10)
nn = NeuralNetwork([hiddenLayer, outputLayer])


# Transform the input data apropriately:
trainingInputs = [x.reshape((784,1)) for x in training_set]
#NORMALIZE THEM
trainingInputs = [x/np.linalg.norm(x) for x in trainingInputs]
# Now for the testing data
testing_set = [x.reshape((784,1)) for x in testing_set]
#NORMALIZE
testing_set = [x/np.linalg.norm(x) for x in testing_set]


# Transform the target training data apropriately:
training_targetVectors = [generateVec(x) for x in training_labels]


# Training:
#def train(self,    inputList,       targetList,   alpha,    numRepetitions=1):
chosen_numRepetitions = 6
nn.train(trainingInputs, training_targetVectors, 0.5,   chosen_numRepetitions)



# Testing
#def test(self, inputList, targetList):
accuracy_percentage = nn.test(testing_set, testing_labels)

print("\n\n")
print("Accuracy of the network is:  "+str(accuracy_percentage))


