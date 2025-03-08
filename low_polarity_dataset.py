# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:37:41 2018

@author: aidaz
"""
# low_polarity_dataset.py (Creates TS2 - Low Polarity Dataset)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import string
import unidecode
from nltk.tokenize import WordPunctTokenizer

# Function for text preprocessing
def preprocess_text(text):
    """Preprocesses text by applying case conversion, contraction expansion, punctuation removal, transliteration, and tokenization."""
    contractions = {
        "she's": "she is", "he's": "he is", "I'm": "I am", "you're": "you are",
        "it's": "it is", "that's": "that is", "what's": "what is", "where's": "where is",
        "who's": "who is", "can't": "cannot", "won't": "will not", "n't": " not",
        "'ll": " will", "'ve": " have", "'re": " are", "'d": " would"
    }
    text = text.lower().strip()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'|'.join(map(re.escape, contractions.keys())), lambda x: contractions[x.group()], text)  # Expand contractions
    text = re.sub(r'[^a-zA-Z0-9.,!?_+/#\s]', '', text)  # Remove unnecessary punctuation
    text = re.sub(r'\.{3,}', ' ', text)  # Replace consecutive dots (more than 2) with a space
    text = unidecode.unidecode(text)  # Transliteration (Unicode to Roman letters)
    text = ' '.join(WordPunctTokenizer().tokenize(text)).strip()  # Tokenization
    return text

# Load original dataset
df = pd.read_csv('comments-valence-3.csv.gz', header=0)

# Select only low polarity (-3, -2 for negative, 2, 3 for positive)
df = df[df['senti.max.autotime'].isin([-3, -2, 2, 3])]

# Convert to binary class labels (Negative → 0, Positive → 1)
df['class'] = np.where(df['senti.max.autotime'] >= 4, 1, 0)

# Remove duplicates and missing values
df = df.drop_duplicates(subset='text', keep='first')
df = df.dropna(subset=['text'])

# Apply text preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Match the size of TS1 to ensure fair evaluation
test_ts1 = pd.read_csv('test_TS1.csv')  # Load TS1 test set to get size reference
test_ts2 = df.sample(n=len(test_ts1), random_state=42)

# Save TS2 dataset
test_ts2.to_csv('test_TS2.csv', index=False)
print("Low polarity dataset (TS2) created and saved.")
