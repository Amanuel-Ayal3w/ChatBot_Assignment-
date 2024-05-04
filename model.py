import torch
import torch.nn as nn

# This part imports the PyTorch library, which is used for building neural networks.
# It also imports the neural network module from PyTorch.

class NeuralNet(nn.Module):
    # This defines a special program called NeuralNet.
    # It's like a blueprint for a neural network.

    def __init__(self, input_size, hidden_size, num_classes):
        # This part sets up the program before using it.

        super(NeuralNet, self).__init__()
        # It sets up the program to use the functions and structure of the nn.Module class.

        self.l1 = nn.Linear(input_size, hidden_size) 
        # This creates the first layer of the neural network.
        # It's like the first step in the network's thinking process.

        self.l2 = nn.Linear(hidden_size, hidden_size) 
        # This creates the second layer of the neural network.
        # It's like the second step in the network's thinking process.

        self.l3 = nn.Linear(hidden_size, num_classes)
        # This creates the third layer of the neural network.
        # It's like the final step in the network's thinking process.

        self.relu = nn.ReLU()
        # This sets up a special math function called ReLU.
        # ReLU helps the network learn and make decisions.

    def forward(self, x):
        # This part is where the main work happens.
        # It's like the actual thinking or processing of the network.

        out = self.l1(x)
        # This sends the input data through the first layer of the network.
        # It's like the input going through the first step of thinking.

        out = self.relu(out)
        # This applies the ReLU function to the output of the first layer.
        # ReLU helps the network decide which information is important.

        out = self.l2(out)
        # This sends the output of the first layer through the second layer.
        # It's like the processed information going through the next step of thinking.

        out = self.relu(out)
        # This applies ReLU again to the output of the second layer.
        # ReLU helps the network make better decisions.

        out = self.l3(out)
        # This sends the output of the second layer through the third layer.
        # It's like the final decision or output of the network.

        return out
        # This gives out the final result or prediction of the network.

# In simple words, this code sets up a program that acts like a brain.
# It takes some input data, processes it through layers of thinking, and gives out a final result.
# This program can be used for tasks like recognizing objects in images or predicting numbers in data.
