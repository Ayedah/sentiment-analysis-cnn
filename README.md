Sentiment Analysis using Multi-Headed CNN

This project implements a multi-headed CNN for sentiment analysis of developer comments, leveraging word embeddings (GloVe + SO Vectors) and POS embeddings.

📂 Folder Structure

📁 sentiment-analysis-cnn
 ├── Create_matrices.py      # Prepares word and POS embeddings
 ├── NeuralNet.py            # Defines and trains the CNN model
 ├── Evaluate_model.py       # Evaluates the model on TS1 & TS2 datasets
 ├── tokenizer.pkl           # Saved tokenizer for consistent text processing
 ├── Multiheaded_CNN.json    # Saved model architecture
 ├── Multiheaded_CNN.h5      # Saved model weights
 ├── SavedData.h5            # Preprocessed embeddings and sequences
 ├── test_TS1.csv            # High-polarity test dataset
 ├── test_TS2.csv            # Low-polarity test dataset
 ├── evaluation_plots.png    # Precision-Recall & ROC curves
 └── README.md               # Project documentation

🛠️ Setup Instructions

1️⃣ Install Dependencies

pip install tensorflow keras pandas numpy gensim spacy h5py scikit-learn matplotlib

2️⃣ Run Preprocessing

python Create_matrices.py

This script generates:

Tokenized sequences

Word embeddings (GloVe + SO Vectors fallback)

POS embeddings (One-hot encoding)

Saved data in SavedData.h5

3️⃣ Train the Model

python NeuralNet.py

Trains the multi-headed CNN using word & POS embeddings

Saves trained model (Multiheaded_CNN.json & .h5)

4️⃣ Evaluate the Model

python Evaluate_model.py

Loads test datasets (TS1 & TS2)

Tokenizes & extracts POS tags

Computes accuracy, precision, recall, F1-score, and confusion matrix

Plots Precision-Recall & ROC curves (evaluation_plots.png)

📊 Model Details

Embeddings:

GloVe: General word embeddings

SO Vectors: SE domain-specific embeddings (fallback mechanism)

POS Embeddings: One-hot vectors based on spaCy POS tags

Model Architecture:

Two embedding inputs (word + POS)

Two 1D convolutional layers (Kernel size = 2, 100 filters each)

Batch normalization & ReLU activation

Global MaxPooling

Dense & dropout layers (0.25, 0.10)

Final output layer with Sigmoid activation

Optimizer: AdaMax

📌 Notes

Tokenizer consistency: Ensure tokenizer.pkl is used for all dataset splits

Data Leakage Avoidance: TS1 & TS2 were not used in embedding creation

Class Imbalance Handling: Training dataset used weighted classes
