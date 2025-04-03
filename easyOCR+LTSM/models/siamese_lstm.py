import keras
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
import keras.backend as K
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Load dataset
dataset_path = "Dataset_Python_Question_Answer.csv"
df = pd.read_csv(dataset_path)

# Prepare tokenizer
text_data = df["Question"].tolist() + df["Answer"].tolist()

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(text_data)

# Save tokenizer
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle)

# Define model parameters
max_len = 100
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

# Define LSTM model
def build_siamese_lstm():
    input_a = Input(shape=(max_len,))
    input_b = Input(shape=(max_len,))

    embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_len)
    lstm_layer = LSTM(64)

    encoded_a = lstm_layer(embedding_layer(input_a))
    encoded_b = lstm_layer(embedding_layer(input_b))

    def euclidean_distance(vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def euclidean_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    distance = Lambda(euclidean_distance, output_shape=euclidean_output_shape)([encoded_a, encoded_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model

# Train and save model
model = build_siamese_lstm()
model.summary()

# Save model
model.save("siamese_lstm_model.h5")
print("✅ Model saved successfully!")
