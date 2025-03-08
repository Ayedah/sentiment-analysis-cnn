# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:13:26 2018

@author: aidaz
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
import spacy
import h5py
import pickle

# Load training and validation datasets
df_train = pd.read_csv("train_TS1.csv")
df_val = pd.read_csv("val_TS1.csv")

df = pd.concat([df_train, df_val], ignore_index=True)  # Combine train and validation sets for embedding creation
text = df['clean_text'].tolist()
labels = df['class'].values

# Tokenization and sequence padding
tokenizer = Tokenizer(filters='$%&()*/:;<=>@[\]^`{|}~\t\n')
tokenizer.fit_on_texts(text)

# Save tokenizer for consistent word mappings
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(text)
maxSeqLength = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=maxSeqLength, padding='pre')

# Load pre-trained embeddings
filepath_glove = 'glove.twitter.27B.200d.txt'
word_vect_glove = KeyedVectors.load_word2vec_format(filepath_glove, binary=False)
word_vect_so = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)

# Create word embedding matrix
embedding_matrix = np.zeros((vocab_size, 200))
for word, index in tokenizer.word_index.items():
    if index >= vocab_size:
        continue
    try:
        embedding_matrix[index] = word_vect_so[word]  # First try SO Vectors (SE domain)
    except KeyError:
        try:
            embedding_matrix[index] = word_vect_glove[word]  # Fallback to GloVe if not found
        except KeyError:
            embedding_matrix[index] = np.random.uniform(-0.25, 0.25, 200)  # Random initialization if missing

# POS Tagging using spaCy
nlp = spacy.load("en_core_web_sm")
vocabulary = ['NAN', 'NIL', 'ADD', 'AFX', 'BES', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'HVS', 'HYPH', 'IN',
              'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
              'PRP$', 'RB', 'RBR', 'RBS', 'RP', '_SP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
              'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', ',', '.', ':', "''", '""', '#', '``', '$', '-LRB-', '-RRB-']

# One-hot encoding for POS tags
embedding_matrix_pos = np.zeros((len(vocabulary), len(vocabulary)))
for i, tag in enumerate(vocabulary):
    embedding_matrix_pos[i, i] = 1

stringToInt = {tag: i for i, tag in enumerate(vocabulary)}

# Convert sentences into POS sequences
embedding_pos = []
for tweet in text:
    t = nlp(tweet)
    pos_tags = [stringToInt.get(token.tag_, 0) for token in t]
    pos_padded = [0] * (maxSeqLength - len(pos_tags)) + pos_tags  # Pre-padding
    embedding_pos.append(pos_padded)
embedding_pos = np.array(embedding_pos)

# Save matrices in HDF5 format
with h5py.File('SavedData.h5', 'w') as hf:
    hf.create_dataset("sequence_data", data=data)
    hf.create_dataset("data_Matrix", data=embedding_matrix)
    hf.create_dataset("sequence_POS", data=embedding_pos)
    hf.create_dataset("POS_Matrix", data=embedding_matrix_pos)
    hf.create_dataset("Labels", data=labels)
    hf.create_dataset("maxSeqLength", data=maxSeqLength)
    hf.create_dataset("vocab_size", data=vocab_size)

print("Embeddings and tokenized sequences saved successfully.")
