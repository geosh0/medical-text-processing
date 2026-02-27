# eda_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import WordPunctTokenizer
import string
import nltk
from nltk.corpus import stopwords

# Ensure resources are present for EDA
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def plot_positional_bias(df, limit=15):
    """Plots the probability of each label based on sentence position."""
    position_bias = pd.crosstab(df['sentence_id'], df['label'], normalize='index')
    position_bias_limited = position_bias.iloc[:limit]

    position_bias_limited.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    plt.title('Probability of Each Label based on Sentence Position', fontsize=16)
    plt.xlabel('Sentence Position (0 = First sentence)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_label_distribution(df, title='Label Distribution'):
    """Plots the count of each label."""
    label_counts = df['label'].value_counts()
    plt.figure(figsize=(10, 7))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.show()

def plot_sentence_lengths(df):
    """Plots distribution of character and word counts."""
    # Calc lengths if not present
    if 'char_length' not in df.columns:
        df['char_length'] = df['text'].apply(len)
    if 'word_count' not in df.columns:
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['char_length'], bins=50, kde=True)
    plt.title('Sentence Length (Characters)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['word_count'], bins=50, kde=True)
    plt.title('Sentence Length (Words)')
    
    plt.tight_layout()
    plt.show()

def plot_top_words(df, top_n=20):
    """Tokenizes raw text, filters basic stopwords/punctuation and plots top words."""
    tokenizer = WordPunctTokenizer()
    stop_words = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)
    punctuation_set.remove('@') # Keep @

    def quick_tokenize(text):
        tokens = tokenizer.tokenize(text.lower())
        return [t for t in tokens if t not in punctuation_set and t not in stop_words]

    print("Tokenizing for EDA (this may take a moment)...")
    all_tokens = [t for text in df['text'] for t in quick_tokenize(text)]
    counter = Counter(all_tokens)
    most_common = counter.most_common(top_n)

    words = [w for w, c in most_common]
    counts = [c for w, c in most_common]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=words, y=counts)
    plt.title(f'Top {top_n} Common Words (Stopwords Removed)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_transition_matrix(df):
    """Plots the label transition probability heatmap."""
    df_sorted = df.sort_values(['abstract_id', 'sentence_id'])
    df_sorted['previous_label'] = df_sorted.groupby('abstract_id')['label'].shift(1)
    transitions = df_sorted.dropna(subset=['previous_label'])

    transition_matrix = pd.crosstab(transitions['previous_label'], transitions['label'], normalize='index')

    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
    plt.title('Label Transition Probability Matrix', fontsize=16)
    plt.show()