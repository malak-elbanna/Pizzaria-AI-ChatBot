
# Pizzaria AI ChatBot

The idea behind this project is an AI Chatbot that can be used to enhance the customer experience on a hypothetical e-commerce website. E-commerce is the idea of buying and selling through electronic commerce techniques, which is basically what the hypothetical Pizzaria does to sell their pizza. That’s why the training data of the AI model is all about delivery, flavors, etc! 

The AI Chatbot is built using specific built-in libraries such as PyTorch, NLTK, NumPy, and JSON. Specifically, our chatbot gets a message from the user and then responds to this message based on the training data provided within the coding process. The chatbot can handle situations starting from "Hi" to "How much time does home delivery take?". It would provide the user with a detailed view of the items they can buy and choose from. Users can also quit the bot at any time if they’re done using it. The bot also supports generating an OTP code that is unique for each user and can be used to confirm the payment process; it’s sent to the email address provided by the user. Additionally, the project is supported by a simple GUI built using Tkinter.

This project was an opportunity for me to showcase my skills in Python as well as a gateway into basic natural language processing and machine learning concepts.


## Installation

### Create a virtual environment

Run the following in your terminal
```bash
  mkdir my-project
  cd my-project
  python3 -m venv venv
```

### Activate the virtual env

Windows
```bash
  venv\Scripts\activate
```

### Dependancies Installation
PyTorch
```bash
  pip install torch
```
NLTK
```bash
  pip install nltk
```
#### run this command if you're unsure you have 'punkt' or not
```bash
  python
  >>> import nltk
  >>> nltk.download('punkt')
```
NumPy
```bash
  pip install numpy
```


## GUI dependencies
Note that you don't need to install any package for the GUI. That's because "tkinter" comes along with Python.

#### Important: If you don't have Python installed on your PC, download the latest version from https://www.python.org/downloads/

## Run

First, run the train.py file. You can run it through your text editor or type this command on your terminal:
```bash
  python train.py
```
Second, run the chat.py file. You can run it through your text editor or type this command on your terminal:
```bash
  python chat.py
```
For the GUI version, run the gui.py file. You can run it through your text editor or type this command on your terminal:
```bash
  python gui.py
```
