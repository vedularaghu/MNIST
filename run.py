import MNIST_loader as load
import NeuralNetwork as nn
import sys

# print(str(sys.argv[1]))
# print(str(sys.argv[2]))
# print(str(sys.argv[3]))

training_data, validation_data, test_data = load.load_data_wrapper()
net = nn.Network([784, 30, 10])
#print(type(training_data))
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)