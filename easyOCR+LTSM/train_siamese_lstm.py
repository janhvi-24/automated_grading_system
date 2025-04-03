import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from models.siamese_lstm import create_siamese_lstm_model

# Load dataset
DATASET_PATH = "Dataset_Python_Question_Answer.csv"  # Ensure this is correct
df = pd.read_csv(DATASET_PATH)

# Display dataset columns
print("Available Columns in Dataset:", df.columns.tolist())

# Verify required columns
required_columns = ['Question', 'Answer']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"⚠️ Error: Missing columns in dataset: {missing_columns}")
    exit(1)  # Stop execution if required columns are missing

# Drop missing values
df = df.dropna(subset=required_columns)

# Generate dummy labels (as dataset lacks an explicit 'Label' column)
df['Label'] = np.ones(len(df))  # Assuming all pairs are correct (1)

# Tokenization parameters
VOCAB_SIZE = 5000
MAX_LENGTH = 100
EMBEDDING_DIM = 128

# Tokenize questions and answers
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Question'].tolist() + df['Answer'].tolist())

# Save tokenizer for inference
TOKENIZER_PATH = "models/tokenizer.pkl"
with open(TOKENIZER_PATH, "wb") as handle:
    pickle.dump(tokenizer, handle)

print(f"✅ Tokenizer successfully saved at {TOKENIZER_PATH}")

# Convert text to sequences
q_seq = tokenizer.texts_to_sequences(df['Question'])
a_seq = tokenizer.texts_to_sequences(df['Answer'])

# Padding sequences
q_padded = pad_sequences(q_seq, maxlen=MAX_LENGTH, padding='post')
a_padded = pad_sequences(a_seq, maxlen=MAX_LENGTH, padding='post')

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    list(zip(q_padded, a_padded)), df['Label'].values, test_size=0.2, random_state=42
)

# Separate inputs for training
X_train_q, X_train_a = zip(*X_train)
X_test_q, X_test_a = zip(*X_test)

X_train_q, X_train_a = np.array(X_train_q), np.array(X_train_a)
X_test_q, X_test_a = np.array(X_test_q), np.array(X_test_a)

# Create model
model = create_siamese_lstm_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)

# Train model
model.fit([X_train_q, X_train_a], y_train, validation_data=([X_test_q, X_test_a], y_test), epochs=10, batch_size=32)

# Save the trained model
MODEL_PATH = "models/siamese_lstm_model.h5"
model.save(MODEL_PATH)
print(f"✅ Model successfully saved at {MODEL_PATH}")
