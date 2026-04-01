from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, SpatialDropout1D,
    Conv1D, MaxPooling1D, GlobalMaxPooling1D
)

def build_lstm_model(num_words=10000, maxlen=500, embedding_dim=128, lstm_units=64):
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen),
        SpatialDropout1D(0.2),
        LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(num_words=10000, maxlen=500, embedding_dim=128):
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen),
        Conv1D(32, 7, activation='relu'),
        MaxPooling1D(5),
        Conv1D(32, 7, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
