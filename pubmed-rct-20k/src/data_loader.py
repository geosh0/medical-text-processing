# data_loader.py
import pandas as pd
from datasets import load_dataset

def load_pubmed_data(dataset_name="armanc/pubmed-rct20k"):
    """
    Loads the PubMed RCT dataset from HuggingFace and converts 
    train/test splits to Pandas DataFrames.
    """
    print(f"Loading dataset: {dataset_name}...")
    pubmed_dataset = load_dataset(dataset_name)

    # Access splits
    train_ds = pubmed_dataset['train']
    test_ds = pubmed_dataset['test']

    # Convert to Pandas
    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df