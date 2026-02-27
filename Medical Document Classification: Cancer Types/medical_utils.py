# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
import os
import re
import string
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder

# Download NLTK resources silently
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ==========================================
# 2. CONFIGURATION & CONSTANTS
# ==========================================
class MedicalConfig:
    """Central configuration for paths, regex patterns, and stopwords."""
    
    # Artifact Mapping (Fixing encoding errors)
    ARTIFACT_MAP = {
        'ï¬': 'fi', 'ï¬‚': 'fl', 'ï': 'i', '¬': '',
        '“': '"', '”': '"', '‘': "'", '’': "'",
        'ˆ': '^', '—': '-', '–': '-', '±': ' ', 
        'µ': 'u', '²': '2', 'Ã': 'A', 'Â': 'A', 
        '™': '', '¼': '1/4', '‰': '%', 'î': 'i', '\xa0': ' '
    }

    # Boilerplate Phrases (The list you provided, pre-compiled)
    BOILERPLATE_LIST = [
        "creative commons", "open access", "rights reserved", "plos one", "material included", 
        "supplementary material", "et al", "credit line", "additional file", "included licence", 
        "distribution reproduction", "attribution international", "third party", "august plos", 
        "authors access", "images third", "sharing adaptation", "adaptation distribution", "full text",
        "copy of this license", "visit http", "org licenses", "permits distribution", "authors source", 
        "format long", "submit your", "manuscript here", "permit distribution", "source provide", 
        "provide link", "permitted statutory", "statutory regulation", "regulation exceeds", 
        "copyright holder", "otherwise material", "permitted need", "credit link", "made image",
        "uplcesiqtofmsms", "sun sun", "unless indicated", "directly view", "link licence", 
        "press limited", "information available", "view copy", "otherwise stated", "copy licence", 
        "licence intended", "credit indicate", "public domain", "unless otherwise", "directly licence", 
        "dedication waiver", "made available", "behalf biochemical", "biochemical society", 
        "society distributed", "published portland", "available unless", "published via", 
        "obtain permission", "image material", "exceeds need", "reproduction format", "material material"
    ]
    BOILERPLATE_PATTERN = re.compile(r'\b(' + '|'.join(BOILERPLATE_LIST) + r')\b', re.IGNORECASE)

    @staticmethod
    def get_stop_words():
        """Returns the consolidated list of standard + domain specific stopwords."""
        stops = set(stopwords.words('english'))
        
        # Domain specific (Academic & Medical)
        domain_stops = {
            'study', 'studies', 'used', 'using', 'results', 'data', 'table', 'figure', 'fig',
            'patients', 'analysis', 'may', 'associated', 'background', 'methods', 'patient',
            'conclusion', 'result', 'method', 'also', 'however', 'significant', 
            'respectively', 'shown', 'vol', 'pp', 'page', 'http', 'www', 'doi', 
            'journal', 'volume', 'issue', 'license', 'august', 'one', 'two',
            'access', 'directly', 'permitted', 'intended', 'indicate', 'attribution', 
            'licensed', 'rights', 'reserved', 'permission', 'use', 'commercial', 
            'attributed', 'medium', 'provided', 'original', 'author',
            'cell', 'cells', 'expression', 'tumor', 'cancer', 'gene', 'genes', 
            'group', 'treatment', 'clinical', 'disease', 'protein', 'levels',
            'activity', 'human', 'sample', 'samples', 'control', 'compared',
            'plusminus', 'found', 'showed', 'high', 'breast', 'reported', 
            'increased', 'different', 'significantly',
            'commonslicence', 'material', 'included', 'distribution'
        }
        stops.update(domain_stops)
        return stops

# ==========================================
# 3. DATA LOADER CLASS
# ==========================================
class DataLoader:
    def load_data(self):
        """Handles KaggleHub download and encoding fallbacks."""
        print("--- 1. Loading Data ---")
        try:
            path = kagglehub.dataset_download("falgunipatel19/biomedical-text-publication-classification")
            csv_path = os.path.join(path, "alldata_1_for_kaggle.csv")
        except:
            csv_path = "alldata_1_for_kaggle.csv"
            print("KaggleHub failed, looking for local file...")

        # Load with encoding check
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, falling back to latin-1...")
            df = pd.read_csv(csv_path, encoding='latin-1')

        # Basic Rename & Drop
        df.columns = ["Serial Number", "Class Labels", "text"]
        if 'Serial Number' in df.columns:
            df.drop('Serial Number', axis=1, inplace=True)
            
        print(f"Initial Shape: {df.shape}")
        
        # Deduplication
        df = df.dropna(subset=['text']).drop_duplicates(subset=['text'], keep='first')
        print(f"Clean Shape (No Dups/NaNs): {df.shape}")
        return df

# ==========================================
# 4. TEXT PREPROCESSOR CLASS
# ==========================================
class TextPreprocessor:
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = MedicalConfig.get_stop_words()
        
    def clean_text(self, text):
        """Applies full cleaning pipeline: Artifacts -> Boilerplate -> Lemma."""
        if not isinstance(text, str): return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Artifact Replacement
        for garbage, fix in MedicalConfig.ARTIFACT_MAP.items():
            text = text.replace(garbage, fix)

        # 3. Boilerplate Removal
        text = MedicalConfig.BOILERPLATE_PATTERN.sub(" ", text)

        # 4. Specific Fixes
        text = re.sub(r'identifi\s+ed', 'identified', text)
        text = re.sub(r'fi\s+rst', 'first', text)
        
        # 5. Remove Numbers & Punctuation
        text = re.sub(r'\b\d+\b', '', text) 
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # 6. Tokenize & Lemmatize
        tokens = self.tokenizer.tokenize(text)
        clean_tokens = [
            self.lemmatizer.lemmatize(t) for t in tokens 
            if len(t) > 2 and t.isalpha() and t not in self.stop_words
        ]
        
        # Double check stop words after lemmatization (e.g. 'cells' -> 'cell')
        final_tokens = [t for t in clean_tokens if t not in self.stop_words]
        
        return " ".join(final_tokens)

# ==========================================
# 5. VISUALIZATION CLASS (Helper)
# ==========================================
class MedicalVisualizer:
    @staticmethod
    def plot_class_distribution(df, col='Class Labels'):
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df[col], palette='viridis', order=df[col].value_counts().index)
        plt.title('Class Distribution')
        plt.show()

    @staticmethod
    def plot_length_distribution(df, col='text_length', hue='Class Labels'):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], bins=50, kde=True, color='teal')
        plt.title('Document Length Distribution')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=hue, y=col, data=df, palette='viridis')
        plt.title(f'Length by {hue}')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_wordclouds(df, text_col, label_col):
        classes = df[label_col].unique()
        plt.figure(figsize=(20, 8))
        
        for i, label in enumerate(classes):
            subset = df[df[label_col] == label]
            text = " ".join(subset[text_col].astype(str))
            wc = WordCloud(background_color='white', max_words=50, width=800, height=400, colormap='Reds')
            wc.generate(text)
            
            plt.subplot(1, 3, i+1)
            plt.imshow(wc, interpolation='bilinear')
            plt.title(f"{label.upper()}", fontsize=16)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# ==========================================
# 6. MACHINE LEARNING TRAINER
# ==========================================
class MLTrainer:
    """
    Handles training and evaluation of standard Scikit-Learn models.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state, n_jobs=-1),
            "Linear SVM": LinearSVC(random_state=random_state, class_weight='balanced', max_iter=2000, dual='auto')
        }
        self.results = []

    def run_benchmark(self, feature_sets, y_train, y_test, label_names):
        """
        Runs all models across all feature sets (BoW, TF-IDF).
        """
        print(f"\n--- Starting ML Benchmark ---")
        self.results = [] # Reset results

        for feature_name, (X_tr, X_te) in feature_sets.items():
            for model_name, model in self.models.items():
                print(f"Training {model_name} on {feature_name}...")
                
                # Train
                start = time.time()
                model.fit(X_tr, y_train)
                train_time = time.time() - start
                
                # Predict
                y_pred = model.predict(X_te)
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
                
                self.results.append({
                    "Model": model_name,
                    "Features": feature_name,
                    "Accuracy": acc,
                    "F1-Score": report['weighted avg']['f1-score'],
                    "Time (s)": round(train_time, 2)
                })
        
        return pd.DataFrame(self.results).sort_values(by="F1-Score", ascending=False)

# ==========================================
# 7. DEEP LEARNING PREPROCESSOR
# ==========================================
class DLPreprocessor:
    """
    Handles OneHotEncoding and TextVectorization for Keras.
    """
    def __init__(self, max_features=7000, sequence_length=500):
        self.max_features = max_features
        self.sequence_length = sequence_length
        self.vectorizer = None
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def fit_transform_labels(self, y_train, y_test):
        """One-hot encodes the target labels."""
        # Reshape to 2D array as required by sklearn
        y_train_enc = self.one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_enc = self.one_hot_encoder.transform(y_test.reshape(-1, 1))
        return y_train_enc, y_test_enc

    def prepare_sequences(self, X_train_text, X_test_text):
        """Fits TextVectorization on Train and transforms both."""
        # Ensure inputs are numpy arrays of strings
        X_train_np = X_train_text.values.astype(str)
        X_test_np = X_test_text.values.astype(str)

        self.vectorizer = TextVectorization(
            max_tokens=self.max_features + 2,
            output_mode="int",
            output_sequence_length=self.sequence_length,
            pad_to_max_tokens=True
        )
        
        print(f"Adapting vectorizer to {len(X_train_np)} documents...")
        self.vectorizer.adapt(X_train_np)
        
        train_seq = self.vectorizer(X_train_np).numpy()
        test_seq = self.vectorizer(X_test_np).numpy()
        
        return train_seq, test_seq

    @property
    def vocab_size(self):
        if self.vectorizer:
            return len(self.vectorizer.get_vocabulary())
        return 0

# ==========================================
# 8. DEEP LEARNING MODEL FACTORY
# ==========================================
class DeepLearningFactory:
    """
    Creates compiled Keras models (LSTM, CNN, etc.).
    """
    @staticmethod
    def build_lstm(vocab_size, embedding_dim=32, num_classes=3):
        model = models.Sequential([
            layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
            layers.LSTM(32),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def build_cnn(vocab_size, embedding_dim=32, seq_length=500, num_classes=3):
        model = models.Sequential([
            layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length),
            layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.6),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def plot_history(history, title="Model History"):
        """Plots accuracy and loss over epochs."""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title(f'{title} Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(f'{title} Loss')
        plt.legend()
        plt.show()