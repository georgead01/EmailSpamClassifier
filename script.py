import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

# set constants
vocab_size = 10000
embed_dim = 16

# get data
data = pd.read_csv("/Users/georgead/Documents/Projects/SpamHam/spam_ham_dataset.csv")

emails = data['text'].values.tolist()
labels = data['label_num'].values.tolist()

# get training and testing sets
training_size = len(data)//2
training_emails = emails[:training_size]
testing_emails = emails[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

# tokenize data
tokenizer = Tokenizer(num_words= vocab_size, oov_token= '<OOV>')
tokenizer.fit_on_texts(training_emails)
sequences = tokenizer.texts_to_sequences(emails)
padded = pad_sequences(sequences, padding= 'post')

training_padded = padded[:training_size]
testing_padded = padded[training_size:]

input_length = len(padded[0])

# build model
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embed_dim, input_length= input_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation= 'relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

# train model
num_epochs = 30
model.fit(np.array(training_padded), 
          np.array(training_labels), 
          epochs= num_epochs, 
          validation_data= (np.array(testing_padded), np.array(testing_labels)), 
          verbose= 2
          )

