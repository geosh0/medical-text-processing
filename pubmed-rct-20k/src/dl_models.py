# dl_models.py
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

def build_lstm(vocab_size, embedding_dim, output_classes=5):
    """Standard LSTM Model."""
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        layers.LSTM(64, return_sequences=True),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_bilstm(vocab_size, embedding_dim, output_classes=5):
    """Bidirectional LSTM Model."""
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_hybrid_model(vocab_size, embedding_dim, seq_len, output_classes=5):
    """Tribrid Model: Text + Line Number + Total Lines."""
    
    # 1. Text Branch
    text_inputs = layers.Input(shape=(seq_len,), dtype="int64", name="text_input")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(text_inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.GlobalMaxPooling1D()(x)
    text_branch = layers.Dense(64, activation="relu")(x)
    
    # 2. Line Number Branch
    line_num_inputs = layers.Input(shape=(15,), name="line_num_input")
    line_num_branch = layers.Dense(32, activation="relu")(line_num_inputs)
    
    # 3. Total Lines Branch
    total_lines_inputs = layers.Input(shape=(20,), name="total_lines_input")
    total_lines_branch = layers.Dense(32, activation="relu")(total_lines_inputs)
    
    # 4. Merge
    combined = layers.Concatenate()([text_branch, line_num_branch, total_lines_branch])
    
    # 5. Output
    z = layers.Dense(64, activation="relu")(combined)
    z = layers.Dropout(0.3)(z)
    outputs = layers.Dense(output_classes, activation="softmax")(z)
    
    model = Model(inputs=[text_inputs, line_num_inputs, total_lines_inputs], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model