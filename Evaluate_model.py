# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:22:47 2018

@author: aidaz
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
import pickle
from tensorflow.keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import spacy

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load TS1 dataset
ts1_df = pd.read_csv("test_TS1.csv")

# Tokenize and pad TS1 data
sequences_ts1 = tokenizer.texts_to_sequences(ts1_df['clean_text'].tolist())
X_ts1_words = pad_sequences(sequences_ts1, maxlen=maxSeqLength, padding='pre')

# Load TS2 dataset
ts2_df = pd.read_csv("test_TS2.csv")

# Tokenize and pad TS2 data
sequences_ts2 = tokenizer.texts_to_sequences(ts2_df['clean_text'].tolist())
X_ts2_words = pad_sequences(sequences_ts2, maxlen=maxSeqLength, padding='pre')

# Load POS vocabulary & process POS sequences
nlp = spacy.load("en_core_web_sm")
vocabulary = ['NAN', 'NIL', 'ADD', 'AFX', 'BES', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'HVS', 'HYPH', 'IN',
              'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
              'PRP$', 'RB', 'RBR', 'RBS', 'RP', '_SP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
              'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', ',', '.', ':', "''", '""', '#', '``', '$', '-LRB-', '-RRB-']
stringToInt = {tag: i for i, tag in enumerate(vocabulary)}

# Process TS1 POS sequences
ts1_pos = []
for tweet in ts1_df['clean_text'].tolist():
    pos_tags = [stringToInt.get(token.tag_, 0) for token in nlp(tweet)]
    ts1_pos.append([0] * (maxSeqLength - len(pos_tags)) + pos_tags)  # Pre-padding
X_ts1_pos = np.array(ts1_pos)

# Process TS2 POS sequences
ts2_pos = []
for tweet in ts2_df['clean_text'].tolist():
    pos_tags = [stringToInt.get(token.tag_, 0) for token in nlp(tweet)]
    ts2_pos.append([0] * (maxSeqLength - len(pos_tags)) + pos_tags)  # Pre-padding
X_ts2_pos = np.array(ts2_pos)

# Extract labels
y_ts1 = ts1_df['class'].values
y_ts2 = ts2_df['class'].values

# Load trained model
with open("Multiheaded_CNN.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("Multiheaded_CNN.h5")

# Compile model
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])

# Evaluate on TS1
loss_ts1, accuracy_ts1 = model.evaluate([X_ts1_words, X_ts1_pos], y_ts1, verbose=1)
y_pred_ts1_probs = model.predict([X_ts1_words, X_ts1_pos])
y_pred_ts1 = (y_pred_ts1_probs > 0.5).astype(int)

print(f"TS1 Test Accuracy: {accuracy_ts1 * 100:.2f}%")
print("TS1 Classification Report:\n", classification_report(y_ts1, y_pred_ts1))
print("TS1 Confusion Matrix:\n", confusion_matrix(y_ts1, y_pred_ts1))

# Evaluate on TS2
loss_ts2, accuracy_ts2 = model.evaluate([X_ts2_words, X_ts2_pos], y_ts2, verbose=1)
y_pred_ts2_probs = model.predict([X_ts2_words, X_ts2_pos])
y_pred_ts2 = (y_pred_ts2_probs > 0.5).astype(int)

print(f"TS2 Test Accuracy: {accuracy_ts2 * 100:.2f}%")
print("TS2 Classification Report:\n", classification_report(y_ts2, y_pred_ts2))
print("TS2 Confusion Matrix:\n", confusion_matrix(y_ts2, y_pred_ts2))

# Generate evaluation plots
plt.figure(figsize=(15, 10))

# Precision-Recall Curve (TS1)
precision_ts1, recall_ts1, _ = precision_recall_curve(y_ts1, y_pred_ts1_probs)
plt.subplot(2, 2, 1)
plt.plot(recall_ts1, precision_ts1, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (TS1)")

# ROC Curve (TS1)
fpr_ts1, tpr_ts1, _ = roc_curve(y_ts1, y_pred_ts1_probs)
roc_auc_ts1 = auc(fpr_ts1, tpr_ts1)
plt.subplot(2, 2, 2)
plt.plot(fpr_ts1, tpr_ts1, label=f'AUC = {roc_auc_ts1:.2f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (TS1)")
plt.legend()

# Precision-Recall Curve (TS2)
precision_ts2, recall_ts2, _ = precision_recall_curve(y_ts2, y_pred_ts2_probs)
plt.subplot(2, 2, 3)
plt.plot(recall_ts2, precision_ts2, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (TS2)")

# ROC Curve (TS2)
fpr_ts2, tpr_ts2, _ = roc_curve(y_ts2, y_pred_ts2_probs)
roc_auc_ts2 = auc(fpr_ts2, tpr_ts2)
plt.subplot(2, 2, 4)
plt.plot(fpr_ts2, tpr_ts2, label=f'AUC = {roc_auc_ts2:.2f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (TS2)")
plt.legend()

plt.tight_layout()
plt.savefig("evaluation_plots.png")
plt.show()

print("Evaluation completed. Accuracy, Loss, Precision-Recall, and ROC Curve saved as 'evaluation_plots.png'.")
