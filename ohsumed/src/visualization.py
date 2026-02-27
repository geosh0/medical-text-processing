import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_raw_mesh_frequencies(df, mesh_column='mesh_terms', top_n=30):
    all_tags = []
    df[mesh_column].dropna().apply(lambda x: all_tags.extend([t.strip() for t in x.split(';')]))
    freq_df = pd.DataFrame(Counter(all_tags).most_common(top_n), columns=['MeSH Term', 'Frequency'])
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=freq_df, x='Frequency', y='MeSH Term', hue='MeSH Term', palette='magma', legend=False)
    plt.title(f'Top {top_n} Raw MeSH Terms', fontsize=15)
    plt.xlabel('Number of Occurrences', fontsize=12)
    plt.ylabel('Raw MeSH Term', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

def plot_label_shift(df, title="Top Clinical Labels After Cleaning"):
    all_clinical = [label for sublist in df['clinical_terms_list'] for label in sublist]
    counts = pd.Series(Counter(all_clinical)).sort_values(ascending=False).head(30)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette='viridis', legend=False)
    plt.title(title, fontsize=15)
    plt.xlabel('Occurrences')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()

def plot_pareto_coverage(cumulative_perc, n_labels_at_80):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(cumulative_perc)), cumulative_perc.values, color='navy')
    plt.axhline(y=0.80, color='red', linestyle='--', label='80% Pareto Cutoff')
    plt.axvline(x=n_labels_at_80, color='orange', linestyle='--', label=f'N={n_labels_at_80}')
    plt.title("Label Coverage: The Pareto Principle at Work")
    plt.legend()
    plt.show()