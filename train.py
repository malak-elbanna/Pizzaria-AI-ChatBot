import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import Neural_Network

with open('intents.json', 'r') as f:
    intents = json.load(f)


all_words = []
tags = []
tokenized_tag = []
# training data
bow_train = []
labels_train = []

# specifying a GPU device to make the work faster
# either the data is transfered between the CPU and supported GPU in PyTorch or between 2 GPUs (cuda)
def GPU():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def tokenization():
    for intent in intents['intents']:
        # extract the values of the keyword tag from intents.json file
        tag = intent['tag']
        # add each tag to the list of tags
        tags.append(tag)
        for pattern in intent['patterns']:
            # extract the values under the keyword pattern in the intents.json file
            # tokenize the pattern using the function in the nltk_utils.py file
            w = tokenize(pattern)
            # add each word to the list of words
            all_words.extend(w)
            # add each word to the list of tokenized words with the tag of this specific word
            # note: the word and the tag are added as a pair
            tokenized_tag.append((w, tag))
    return all_words, tokenized_tag


tokenization()
# specify whether the token is a punctuation mark or not and if it's, then remove from all_words
ignore_punctuation = ['?', '!', '.', ',', ';', '/']
all_words = [stem(w) for w in all_words if w not in ignore_punctuation]
# deletes repeated words and then sort others in alphabetical order
all_words = sorted(set(all_words))
tags = sorted(set(tags))


# print(len(tokenized_tag), "patterns") to see the number of patterns
# print(len(tags), "tags:", tags) to see the number of tags and tags
# print(len(all_words), "unique stemmed words:", all_words) to see the number of unique stemmed words +stemmed all_words list


# preparation of the training data from the extracted data of intents.json file
def prepare_data():
    for (pattern_sentence, tag) in tokenized_tag:
        # appending bag of words for each pattern_sentence in bow_train list
        bow = bag_of_words(pattern_sentence, all_words)
        bow_train.append(bow)
        # append only class labels (index of each tag in tags list) to labels_train list
        label = tags.index(tag)
        labels_train.append(label)
    return bow_train, labels_train


prepare_data()
# storing these lists in a numpy array for faster memory processing
bow_train = np.array(bow_train)
labels_train = np.array(labels_train)


# Hyper-parameters
num_epochs = 6100
batch_size = 32
learning_rate = 0.001
input_size = len(bow_train[0])
hidden_size = 8
output_size = len(tags)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(bow_train)
        self.x_data = bow_train
        self.y_data = labels_train


    # introducing index to the __getitem__ method to get the index of the elements in x_data and y_data
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # calling len(dataset) function to return the number of samples
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


model = Neural_Network(input_size, hidden_size, output_size).to(GPU())


# loss & optimizer
crcriterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training loop
def training_loop():
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(GPU())
            labels = labels.to(dtype=torch.long).to(GPU())

            # forward pass
            outputs = model(words)
            loss = crcriterion(outputs, labels)

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if (epoch+1) % 100 == 0:
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # print(f'final loss: {loss.item():.4f}')


training_loop()


# saving data on a file in torch
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

# print(f'training complete. file saved to {FILE}')


def OTP_verify():
    import random
    import smtplib

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    password = 'mauayacguiixbwni'
    server.login('malak.a.elbanna@gmail.com', password)
    otp = ''.join([str(random.randint(0, 9)) for i in range(4)])
    msg = 'Hello, your OTP is '+str(otp)
    mail = input('enter your gmail address: ')
    server.sendmail('malak.a.elbanna@gmail.com', mail, msg)
    server.quit()