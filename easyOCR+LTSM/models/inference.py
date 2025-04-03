import pickle
import numpy as np
import keras
from keras.models import load_model
from keras import backend as K

# Load Tokenizer
with open("D:/Btech/SEM-04/ML/Smart Grading system/Final Product/LSTM model + easyOCR/models/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define Custom Functions
@keras.saving.register_keras_serializable()
def euclidean_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

@keras.saving.register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# Load the model with custom objects
model_path = "D:/Btech/SEM-04/ML/Smart Grading system/Final Product/LSTM model + easyOCR/models/siamese_lstm_model.h5"

from keras.losses import MeanSquaredError

model = load_model(model_path, custom_objects={
    "euclidean_distance": euclidean_distance,
    "mse": MeanSquaredError()
})


# model = load_model(model_path, custom_objects={
#     "euclidean_output_shape": euclidean_output_shape,
#     "euclidean_distance": euclidean_distance
# })

# Function to Predict Score
def predict_answer_score(question, answer):
    question_seq = tokenizer.texts_to_sequences([question])
    answer_seq = tokenizer.texts_to_sequences([answer])

    question_pad = keras.preprocessing.sequence.pad_sequences(question_seq, maxlen=100)
    answer_pad = keras.preprocessing.sequence.pad_sequences(answer_seq, maxlen=100)

    score = model.predict([question_pad, answer_pad])[0][0]
    return score
