# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:22:47 2018

@author: aidaz
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalMaxPooling1D, Dropout, Dense, Concatenate
from tensorflow.keras.optimizers import Adamax
import h5py
import numpy as np
import pickle

# Load tokenized sequences and embeddings
with h5py.File('SavedData.h5', 'r') as hf:
    X_words = np.array(hf['sequence_data'])
    X_pos = np.array(hf['sequence_POS'])
    embedding_matrix = np.array(hf['data_Matrix'])
    embedding_matrix_pos = np.array(hf['POS_Matrix'])
    y_labels = np.array(hf['Labels'])
    maxSeqLength = int(np.array(hf['maxSeqLength']))
    vocab_size = int(np.array(hf['vocab_size']))
    pos_vocab_size = embedding_matrix_pos.shape[0]

# Define input layers
word_input = Input(shape=(maxSeqLength,), name='word_input')
pos_input = Input(shape=(maxSeqLength,), name='pos_input')

# Embedding layers (static embeddings)
word_embedding = Embedding(input_dim=vocab_size, output_dim=200,
                           weights=[embedding_matrix], input_length=maxSeqLength,
                           trainable=False, name='word_embedding')(word_input)

pos_embedding = Embedding(input_dim=pos_vocab_size, output_dim=pos_vocab_size,
                          weights=[embedding_matrix_pos], input_length=maxSeqLength,
                          trainable=False, name='pos_embedding')(pos_input)

# Convolutional layers with kernel size 2
word_conv = Conv1D(filters=100, kernel_size=2, name='word_conv')(word_embedding)
pos_conv = Conv1D(filters=100, kernel_size=2, name='pos_conv')(pos_embedding)

# Batch normalization
word_bn = BatchNormalization(name='word_bn')(word_conv)
pos_bn = BatchNormalization(name='pos_bn')(pos_conv)

# Activation layers (ReLU)
word_act = Activation('relu', name='word_act')(word_bn)
pos_act = Activation('relu', name='pos_act')(pos_bn)

# Global max pooling layers
word_pool = GlobalMaxPooling1D(name='word_pool')(word_act)
pos_pool = GlobalMaxPooling1D(name='pos_pool')(pos_act)

# Concatenation of both feature representations
merged = Concatenate(name='concatenate')([word_pool, pos_pool])

# Dropout layers
dropout_1 = Dropout(0.25, name='dropout_1')(merged)
dense_1 = Dense(100, activation='relu', name='dense_1')(dropout_1)
dropout_2 = Dropout(0.10, name='dropout_2')(dense_1)

# Output layer with sigmoid activation
output = Dense(1, activation='sigmoid', name='output')(dropout_2)

# Define model
model = Model(inputs=[word_input, pos_input], outputs=output, name='Multiheaded_CNN')

# Compile model using AdaMax optimizer
model.compile(optimizer=Adamax(), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Save the model architecture
model_json = model.to_json()
with open("Multiheaded_CNN.json", "w") as json_file:
    json_file.write(model_json)

print("Model architecture saved successfully.")
