
# Project Title

This project is for the CSAI101 course. It's an AI ChatBOT that's meant to serve an e-commerce website, specifically a pizza vendor.




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