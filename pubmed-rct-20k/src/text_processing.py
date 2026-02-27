# text_processing.py
import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from sklearn.preprocessing import LabelEncoder

# Ensure resources are present
def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res)

def get_custom_stopwords():
    """Builds the custom stopword list including domain specific noise."""
    stop_words = set(stopwords.words('english'))
    
    # Custom domain words to remove
    domain_stopwords = ['rsb', 'lsb', 'total', 'also', 'however']
    stop_words.update(domain_stopwords)
    
    # Words to KEEP (remove from stopword list)
    valuable_words = ['against', 'between']
    for word in valuable_words:
        if word in stop_words:
            stop_words.remove(word)
            
    return stop_words

def clean_text_initial(df, col_name='text'):
    """
    Applies lowercasing, translation (removing punct but keeping @),
    removes weird symbols and digits.
    """
    download_nltk_resources()
    print("Applying initial text cleaning...")
    
    # 1. Define Translation Table (Protect @)
    punctuation_to_remove = string.punctuation.replace('@', '')
    translator = str.maketrans('', '', punctuation_to_remove)

    # 2. Process
    # Helper function to apply to series
    def clean_str(text):
        # Lowercase and remove punctuation
        t = text.lower().translate(translator)
        return t

    processed = df[col_name].apply(clean_str)

    # 3. Remove weird symbols
    weird_symbols = ['‘', '™', '±', '¬', 'ï', '\u2610', chr(147), chr(145)]
    for symbol in weird_symbols:
        processed = processed.str.replace(symbol, '', regex=False)

    # 4. Remove digits (keep @)
    processed = processed.str.replace(r'\d+', '', regex=True)
    
    # 5. Whitespace
    processed = processed.apply(lambda x: " ".join(x.split()))
    
    df['processed_text'] = processed
    return df

def tokenize_and_lemmatize(df):
    """
    Tokenizes 'processed_text', lemmatizes, removes custom stopwords, 
    and creates 'final_tokens' and 'processed_text_final'.
    """
    print("Tokenizing and Lemmatizing...")
    tokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()
    stop_words = get_custom_stopwords()

    # Tokenize
    df['tokens'] = df['processed_text'].apply(tokenizer.tokenize)

    # Lemmatize & Filter
    def process(tokens):
        return [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]

    df['final_tokens'] = df['tokens'].apply(process)
    
    # Rejoin
    df['processed_text_final'] = df['final_tokens'].apply(lambda x: ' '.join(x))
    
    return df

def get_corpus_stats(df, coverage_target=0.95):
    """
    Analyzes the training dataframe to find suggested max_features and max_length.
    """
    # Vocab Analysis
    all_tokens = [t for tokens in df['final_tokens'] for t in tokens]
    counts = Counter(all_tokens)
    total_tokens = sum(counts.values())
    sorted_vocab = counts.most_common()

    cumulative = 0
    suggested_max_features = 0
    for i, (term, freq) in enumerate(sorted_vocab):
        cumulative += freq
        if (cumulative / total_tokens) >= coverage_target:
            suggested_max_features = i + 1
            break
            
    # Length Analysis
    lengths = df['final_tokens'].apply(len)
    suggested_max_len = int(lengths.quantile(coverage_target))
    
    print(f"--- Corpus Stats (Target Coverage: {coverage_target*100}%) ---")
    print(f"Suggested Vocabulary Size (max_features): {suggested_max_features}")
    print(f"Suggested Sequence Length (max_length): {suggested_max_len}")
    
    return suggested_max_features, suggested_max_len

def encode_labels(train_df, test_df):
    """
    Encodes the target labels. Returns the encoder and transformed arrays.
    """
    le = LabelEncoder()
    train_y = le.fit_transform(train_df['label'])
    test_y = le.transform(test_df['label'])
    
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {mapping}")
    
    return le, train_y, test_y