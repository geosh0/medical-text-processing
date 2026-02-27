# 🩺 OHSUMED: Multi-Label Clinical Text Classification

## 🎯 Objective
The goal of this project is to build a robust multi-label classification system capable of assigning relevant clinical MeSH (Medical Subject Headings) terms to raw medical abstracts. 

Because medical documents frequently belong to multiple categories simultaneously (e.g., a paper discussing both *Heart Disease* and *Diabetes*), this is formulated as a **Multi-Label Classification** problem.

## 🧠 The "What" and the "Why": Pipeline Architecture

Dealing with raw medical text and thousands of potential labels requires strict engineering and data science decisions. Here is the rationale behind the pipeline:

### 1. Smart Label Processing & The Pareto Principle
* **The Problem:** The raw OHSUMED dataset contains tens of thousands of unique MeSH terms, many of which are non-clinical meta-tags (e.g., *"Male"*, *"Human"*, *"United States"*, *"Retrospective Study"*). Furthermore, the label distribution is heavily skewed with a massive long tail.
* **The Solution (The "Why"):** 
  * **Heuristic Filtering:** We implemented a domain-specific filter (`RAW_NOISY_TAGS`) to strip demographic and publication metadata. We want our models to learn *clinical conditions*, not study designs.
  * **Pareto Cutoff:** Training a classifier on 15,000 highly imbalanced labels is computationally wasteful and mathematically unstable. We applied the **80/20 Pareto Principle**, identifying the core subset of labels that account for 80% of all clinical tag assignments. This reduced our target space to the "vital few," drastically improving signal-to-noise ratio.

### 2. Custom Text Preprocessing
* **The Problem:** Standard NLP stopwords (like "the", "and") are insufficient for medical corpora. Words like *"patient"*, *"study"*, *"clinical"*, and *"results"* appear in almost every medical abstract, offering zero predictive value.
* **The Solution (The "Why"):** We defined a custom list of **"Medical Weeds."** By combining standard NLTK stopwords with these domain-specific weeds, and normalizing numeric measurements to a generic `[num]` token, we forced the TF-IDF and deep learning models to focus on actual medical terminology.

### 3. Feature Engineering & Memory Safety
* **The Problem:** Extracting Deep Learning embeddings (like BioBERT) for tens of thousands of documents will instantly cause an Out-Of-Memory (OOM) error on standard hardware.
* **The Solution (The "Why"):** We created a tiered feature pipeline:
  1. **Classical Baselines:** BoW and TF-IDF (1,2-grams) stored efficiently as compressed sparse row matrices (`.npz`).
  2. **Dense Semantic Vectors:** Custom Word2Vec and pre-trained GloVe embeddings.
  3. **BioBERT:** We implemented a **memory-safe extraction loop** using PyTorch's `autocast` (FP16 mixed precision), batched processing, and aggressive garbage collection. This allows us to extract state-of-the-art transformer embeddings on standard GPUs without crashing.

### 4. The Modeling Showdown
* **The Problem:** Multi-label classification requires specific architectural setups and evaluation metrics. Standard accuracy is heavily misleading.
* **The Solution (The "Why"):** 
  * **Classical ML:** We used `OneVsRestClassifier` wrappers around Logistic Regression, LinearSVC, and XGBoost. We implemented "smart skipping" logic to avoid wasting compute (e.g., skipping XGBoost on massive sparse TF-IDF matrices).
  * **Deep Learning:** We built `SimpleRNN`, `LSTM`, and `BiLSTM` networks using TensorFlow/Keras.
  * **Metrics:** Models are evaluated using **Micro/Macro F1-Scores** and **Hamming Loss** (the industry standards for multi-label tasks), ensuring we capture performance across both frequent and rare classes.

## 📂 Directory Structure

To maintain a clean, readable workflow, heavy logic (walls of regex, looping structures, and plot definitions) has been abstracted into the `src/` backend. The Jupyter notebooks are purely linear data narratives.

```text
ohsumed/
│
├── README.md                      <- This document
│
├── src/                           <- Python Backend
│   ├── text_processing.py         <- Regex, stopword lists, and MeSH cleaning
│   ├── feature_extraction.py      <- Memory-safe BioBERT and GloVe extractors
│   ├── ml_models.py               <- Scikit-learn/XGBoost training loops
│   ├── dl_utils.py                <- Keras model builders and chunked evaluators
│   └── visualization.py           <- Matplotlib/Seaborn plotting functions
│
└── ohsumed.ipynb      <- Notebook 2: EDA, Label Filtering, Feature Extraction, ML/DL Model Training & Evaluation Showdown
```
## 🚀 How to Run
Open ohsumed.ipynb and run all cells. This will:
  * download the dataset via HuggingFace,
  * clean the text,
  * calculate the Pareto cutoffs,
  * extract all features (TF-IDF, BERT, etc.), and
  * save them to disk as compressed .npz and .parquet files
  * load the engineered features,
  * run the classical ML showdown,
  * train the deep learning sequence models, and
  * output the final comprehensive evaluation CSVs.
    
## Data from:
* [OHSUMED](https://huggingface.co/datasets/community-datasets/ohsumed)
