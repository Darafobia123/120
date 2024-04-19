#Text Data Preprocessing Lib
import nltk
import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))



# preprocessing the user's input 
def preprocessing_user_input(user_input):
    i1 = nltk.word_tokenize(user_input)
    i2 = get_stem_words(i1,ignore_words)
    i3 = sorted(list(set(i2)))
    bag=[]
    bag_of_words = []
   
    # Input data encoding 
    for word in words:            
        if word in i3:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
  
    return np.array(bag)

# prediction using model

def prediction(user_input):
    i4 = preprocessing_user_input(user_input)
    prediction = model.predict(i4)
    predict_class_label = np.argmax(prediction[0])
    return predict_class_label
#4

# we want to get the bot's response acc. to prediction
def bot_response(user_input):
    predicted_class_label = prediction(user_input)
    predict_class = classes[predicted_class_label]
    print(predict_class)
    for intent in intents["intents"]:
        if intent["tag"] == predict_class:
            bot_response = random.choice(intent["responses"])
            return bot_response











print("Hi I am Stella, How Can I help you?")

while True:
    user_input = input("Type your message here:")
    print("User Input: ", user_input)

    response = bot_response(user_input)
    print("Bot Response: ", response)

