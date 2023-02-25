import random
import json
import torch
from model import Neural_Network
from nltk_utils import bag_of_words, tokenize
from train import GPU, OTP_verify


with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = Neural_Network(input_size, hidden_size, output_size).to(GPU())
# loading the model
model.load_state_dict(model_state)
# used to turn off layers with different behavior
# without this line, the chatbot will give inaccurate answers
model.eval()

bot_name = "Pizzaria"


def pizzaria_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    # reshape numpy array
    X = X.reshape(1, X.shape[0])
    # convert a number array to pytorch tensor
    X = torch.from_numpy(X).to(GPU())

    output = model(X)
    # return the maximum value of all the tensor elements in the output
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    # .item() returns the key-value pair of the dict as a list of tuples
    prob = probs[0][predicted.item()]
    if prob.item() > 0.95:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Sorry! Try asking the question in another form"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)", end='')
    print(" (type 'OTP verification' to get your payment code)")
    while True:
        # msg = the user input
        msg = input("You: ")
        if msg == "quit":
            break

        elif msg == "OTP verification":
            # calls the OTP_verify() function in train.py file
            OTP_verify()
            break

        resp = pizzaria_response(msg)
        print(f"{bot_name}: {resp}")
