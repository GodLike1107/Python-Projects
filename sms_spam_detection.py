import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


# Load Data
def load_data(file_path='data/spam.csv'):
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
        return df['message'].values, df['label'].values
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return None, None


# Text Preprocessing
def preprocess_text(texts, max_words=5000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer


# Optimized Model with Bidirectional LSTM & Dropout
def create_model(vocab_size, max_len=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train and Evaluate Model
def train_and_evaluate():
    texts, labels = load_data()
    if texts is None or labels is None:
        return None, None, None
    X, tokenizer = preprocess_text(texts)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model(vocab_size=5000)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5)

    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr], verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return model, history, tokenizer


# Plot Training History
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Predict New Messages
def predict_spam(model, tokenizer, text, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    label = "Spam" if prediction > 0.5 else "Ham"
    return label, prediction


# Main Execution
def main():
    model, history, tokenizer = train_and_evaluate()
    if model is None:
        return
    plot_history(history)
    test_messages = [
        "Congratulations! You've won a $1000 gift card! Call now!",
        "Hey, are you free this weekend to catch up?",
        "URGENT: Your account needs verification. Click here."
    ]
    print("\nExample Predictions:")
    for message in test_messages:
        label, probability = predict_spam(model, tokenizer, message)
        print(f"Message: {message}")
        print(f"Prediction: {label} (Probability: {probability:.4f})\n")


if __name__ == "__main__":
    main()