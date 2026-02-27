import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score

def batch_densify(sparse_matrix, chunk_size=2000):
    rows, cols = sparse_matrix.shape
    dense_result = np.zeros((rows, cols), dtype='uint8')
    for i in range(0, rows, chunk_size):
        end = min(i + chunk_size, rows)
        dense_result[i:end] = sparse_matrix[i:end].toarray().astype('uint8')
    return dense_result

def build_sequence_model(model_type, vocab_size, num_labels):
    model = Sequential([layers.Embedding(vocab_size, 128)])
    
    if model_type == "Simple RNN":
        model.add(layers.SimpleRNN(128, return_sequences=True))
    elif model_type == "Bidirectional RNN":
        model.add(layers.Bidirectional(layers.SimpleRNN(128, return_sequences=True)))
    elif model_type == "LSTM":
        model.add(layers.LSTM(128, return_sequences=True)) # FIXED!
    elif model_type == "Bidirectional LSTM":
        model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True))) # FIXED!
        
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_labels, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc', multi_label=True)])
    return model

def evaluate_dl_model_memory(model, X_test_data, y_test_data, name, chunk_size=10000):
    num_samples, num_labels = X_test_data.shape[0], y_test_data.shape[1]
    y_pred = np.zeros((num_samples, num_labels), dtype='uint8')
    
    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)
        chunk_prob = model.predict(X_test_data[i:end], batch_size=256, verbose=0)
        y_pred[i:end] = (chunk_prob >= 0.5).astype('uint8')
        if i % (chunk_size * 5) == 0: gc.collect()

    results = {
        "Model": name,
        "Accuracy (Exact Match)": round(accuracy_score(y_test_data, y_pred), 4),
        "Hamming Loss": round(hamming_loss(y_test_data, y_pred), 4),
        "F1 (Micro)": round(f1_score(y_test_data, y_pred, average='micro', zero_division=0), 4),
        "F1 (Macro)": round(f1_score(y_test_data, y_pred, average='macro', zero_division=0), 4)
    }
    del y_pred
    gc.collect()
    return results