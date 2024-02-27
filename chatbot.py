import random
import json
import pickle
import numpy as np
from flask import Flask, request

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'C:\Users\vishal\OneDrive\Desktop\Chatbot code files\intent.json').read())


try:
    with open(r'C:\Users\vishal\OneDrive\Desktop\Chatbot code files\words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open(r'C:\Users\vishal\OneDrive\Desktop\Chatbot code files\classes.pkl', 'rb') as f:
        classes = pickle.load(f)
except Exception as e:
    print("Error loading pickled files:", e)
    exit()

model = load_model(r'C:\Users\vishal\OneDrive\Desktop\Chatbot code files\chatbotmodel.h5')

# print(model)

def clean_up_sentense(sentence):
    sentence_word = nltk.word_tokenize(sentence)
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word

def bag_of_words(sentence):
    sentence_words = clean_up_sentense(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intent_json):
    tag = intents_list[0]['intent']
    list_of_intents = intent_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

print("Go! Bot is running!")

while True:
    message = input("")

    if message.lower() in ['exit', 'quit']:
        print("Closing the bot. Goodbye!")
        break
    
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)



