import random  # Importing the random module for generating random responses
import json  # Importing the json module for handling JSON data
import torch  # Importing the PyTorch library for machine learning

from model import NeuralNet  # Importing the NeuralNet class from a custom module
from nltk_utils import bag_of_words, tokenize  # Importing utility functions for text preprocessing

# Check if GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from a JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained model data from a file
FILE = "data.pth"
data = torch.load(FILE)

# Extract required information from the loaded data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load the model with the pre-trained weights
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"  # Name of the chatbot
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")  # Get user input
    if sentence == "quit":  # If user types 'quit', exit the conversation
        break

    sentence = tokenize(sentence)  # Tokenize the user input
    X = bag_of_words(sentence, all_words)  # Create bag of words representation for the input
    X = X.reshape(1, X.shape[0])  # Reshape the input to match the model input size
    X = torch.from_numpy(X).to(device)  # Convert input to PyTorch tensor and move to device

    output = model(X)  # Get the model's prediction for the input
    _, predicted = torch.max(output, dim=1)  # Get the predicted tag

    # Calculate probability of the predicted tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If predicted probability is above a threshold, select a response from intents and print it
    if prob.item() > 0.9:
        for intent in intents['intents']:
            if tags[predicted.item()] == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:  # If not, print a default response indicating lack of understanding
        print(f"{bot_name}: I do not understand...")
