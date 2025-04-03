

import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer  # Use tensorflow.keras

# Load dataset
dataset_path = "Dataset_Python_Question_Answer.csv"
df = pd.read_csv(dataset_path)

# Extract questions and answers
questions = df['Question'].astype(str).tolist()
answers = df['Answer'].astype(str).tolist()

# Combine text data
text_data = questions + answers

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)

# Save tokenizer
tokenizer_path = "tokenizer.pkl"
with open(tokenizer_path, "wb") as handle:
    pickle.dump(tokenizer, handle)

print(f"✅ Tokenizer successfully saved as {tokenizer_path}")

