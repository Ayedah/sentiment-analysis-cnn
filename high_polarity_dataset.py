
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 22:39:56 2018

@author: aidaz
"""
# high_polarity_dataset.py (Creates TS1 - High Polarity Dataset)

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

# Load dataset
df = pd.read_csv('comments-strong-valence.csv.gz', header=0)

# Select only extreme polarity (-4, -5 for negative, 4, 5 for positive)
df = df[df['senti.max.autotime'].isin([-4, -5, 4, 5])]

# Convert to binary class labels (Negative → 0, Positive → 1)
df['class'] = np.where(df['senti.max.autotime'] >= 4, 1, 0)

# Remove duplicates and missing values
df = df.drop_duplicates(subset='text', keep='first')
df = df.dropna(subset=['text'])

# Apply text preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Balance dataset (70% positive, 30% negative)
pos_samples = df[df['class'] == 1]
neg_samples = df[df['class'] == 0]
pos_sampled = pos_samples.sample(n=615283, random_state=42)
df_balanced = pd.concat([pos_sampled, neg_samples]).sample(frac=1, random_state=42)

# Split into training, validation, and test (TS1) with correct positive-negative ratios
train_pos, test_pos = train_test_split(pos_sampled, test_size=0.2, random_state=42)
train_neg, test_neg = train_test_split(neg_samples, test_size=0.1, random_state=42)

train_pos, val_pos = train_test_split(train_pos, test_size=0.2, random_state=42)
train_neg, val_neg = train_test_split(train_neg, test_size=0.1, random_state=42)

train_data = pd.concat([train_pos, train_neg])
val_data = pd.concat([val_pos, val_neg])
test_ts1 = pd.concat([test_pos, test_neg])

# Save datasets
train_data.to_csv('train_TS1.csv', index=False)
val_data.to_csv('val_TS1.csv', index=False)
test_ts1.to_csv('test_TS1.csv', index=False)
print("High polarity dataset (TS1) created and saved.")
