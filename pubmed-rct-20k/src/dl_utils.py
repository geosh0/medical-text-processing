# dl_utils.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def encode_labels_one_hot(train_y, test_y):
    """One-hot encodes the integer labels."""
    enc = OneHotEncoder(sparse_output=False)
    # Reshape to 2D array (n_samples, 1)
    train_oh = enc.fit_transform(train_y.reshape(-1, 1))
    test_oh = enc.transform(test_y.reshape(-1, 1))
    return train_oh, test_oh

def create_text_vectorizer(train_texts, max_tokens, output_seq_len):
    """Creates and adapts a TextVectorization layer."""
    print(f"\nSetting up TextVectorization (vocab: {max_tokens}, len: {output_seq_len})...")
    text_vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_seq_len,
        standardize=None 
    )
    text_vectorizer.adapt(train_texts)
    return text_vectorizer

def prepare_positional_features(df, split_name="train"):
    """
    Calculates line numbers and total lines, returns one-hot encoded arrays.
    """
    # Calculate Total Lines if not present
    if 'total_lines' not in df.columns:
        df['total_lines'] = df.groupby('abstract_id')['sentence_id'].transform('count')
        
    # One-Hot Encode Line Number (limit 15)
    line_num_oh = tf.one_hot(df['sentence_id'], depth=15).numpy()
    
    # One-Hot Encode Total Lines (limit 20)
    total_lines_oh = tf.one_hot(df['total_lines'], depth=20).numpy()
    
    print(f"Positional features prepared for {split_name}.")
    return line_num_oh, total_lines_oh

def plot_history(history, model_name):
    """Plots accuracy and loss curves."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.show()

def calculate_dl_results(y_true, y_pred, model_name):
    """Calculates metrics for DL models."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return {
        "Model": model_name,
        "Accuracy": acc,
        "F1-Score (Weighted)": f1
    }