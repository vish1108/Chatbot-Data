# Chatbot with Intent Recognition

This repository contains code for a simple chatbot that utilizes intent recognition to understand user queries and respond accordingly. The chatbot is built using Python and employs a training process to generate necessary files for inference.

## Files Included

1. **intent.json**: This file contains predefined intents along with examples of phrases associated with each intent. It serves as the training data for the chatbot.
2. **training.py**: This script is responsible for training the chatbot model using the intent data provided in `intent.json`. Upon execution, it generates three essential files:
   - **classes.pkl**: Pickled file containing the list of intents/classes extracted from `intent.json`.
   - **words.pkl**: Pickled file containing the list of unique words found in the training phrases.
   - **chatbotmodel.h5**: Trained chatbot model saved in HDF5 format.
3. **chatbot.py**: The main script for the chatbot application. It loads the trained model and necessary files (`classes.pkl` and `words.pkl`) to recognize intents and respond to user queries.

## Usage

To use the chatbot:

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Run `training.py` to train the chatbot model and generate required files (`classes.pkl`, `words.pkl`, and `chatbotmodel.h5`).
4. After training, execute `chatbot.py` to start the chatbot interface.
5. Interact with the chatbot by typing your queries and observing the responses.

## Dependencies

- Python 3.x
- Keras (for training and inference)
- numpy
- nltk (Natural Language Toolkit)

Install the required dependencies using `pip`:

