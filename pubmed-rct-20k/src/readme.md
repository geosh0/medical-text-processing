# 🧬 PubMed 20k RCT: Sequential Sentence Classification

## 📖 Project Overview
Reading through medical abstracts can be time-consuming for researchers. This project builds a Natural Language Processing (NLP) model capable of **automatically categorizing sentences** in medical abstracts into their respective roles:
1. **Background**
2. **Objective**
3. **Methods**
4. **Results**
5. **Conclusions**
   
By understanding the **sequence** and **position** of sentences, we moved beyond simple keyword matching to create a context-aware Deep Learning model that achieves **88.1% accuracy**, significantly outperforming classical machine learning baselines.

## 📂 The Dataset: PubMed 20k RCT
The dataset consists of ~20,000 medical abstracts from Randomized Controlled Trials (RCTs).
* **Source**: [PubMed 20k RCT](https://huggingface.co/datasets/armanc/pubmed-rct20k)
* **Structure**: Each sentence is labeled with its section (e.g., "Methods").
* **Pre-normalization**: The dataset creators replaced numbers with the @ symbol (e.g., "efficacy of @ weeks").

## 🔍 Exploratory Data Analysis (EDA) & Insights
Before modeling, we uncovered three critical structural signals:
1. **The "Linear Flow"**: Abstracts follow a strict narrative (Background → Conclusions). They never move backward.
2. **Positional Importance**: Sentence 0 is almost always Background, while Sentence 10+ is usually Results.
3. **Vocabulary Fingerprints**:
     * **Methods**: Dominant words include "randomized", "placebo", "blind".
     * **Results**: Characterized by "p-value", "CI", "significant", and the bigram ('p', '@').

## ⚙️ Data Preprocessing Pipeline
We engineered a custom pipeline to preserve medical context while preparing data for Deep Learning:
### 1. Text Cleaning
* **Preserved @**: We kept the @ symbol as it acts as a placeholder for dosage and time (e.g., "@ mg" vs "@ weeks").
* **Stopwords**: We removed dataset artifacts (rsb, lsb) but kept relational words (between, with) to preserve directionality (e.g., "Better than placebo").

### 2. The "95% Rule"
Instead of guessing hyperparameters, we calculated statistics from the training set:
* **Sequence Length**: Fixed at 55 tokens (covers 95% of sentences).
* **Vocabulary Size**: Fixed at 12,843 words (covers 95% of tokens).

### 3. Feature Engineering (The Breakthrough)
We generated two non-text features to give the model "map coordinates":
* line_number: The specific index of the sentence in the abstract.
* total_lines: The total length of the abstract.

## 🧠 Modeling Strategy
### Phase 1: Machine Learning Baselines
We benchmarked classical algorithms (Naive Bayes, SVM, Random Forest) against a Logistic Regression + TF-IDF pipeline.
* **Result**: ~77.3% Accuracy.
* **Limitation**: The model confused Background and Objective due to vocabulary overlap. It lacked the context of "where" the sentence appeared.

### Phase 2: Deep Learning (LSTM)
We trained a standard Long Short-Term Memory (LSTM) network to capture sequence information.
* **Result**: ~80.0% Accuracy.
* **Limitation**: Hit a performance ceiling. The model understood the text but still struggled with structural ambiguity.

### Phase 3: The "Tribrid" Model (Final Solution)
We designed a custom Hybrid Architecture that accepts three inputs simultaneously:
1. **Token Embeddings** (Text Sequence) → The Content
2. **Line Number Embedding** (Positional) → The Location
3. **Total Lines Embedding** (Structural) → The Context

## 🏆 Results
The **Tribrid Model** successfully broke the performance ceiling, proving that positional embeddings are critical for structured document classification.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Bayes (Baseline)** | 72.1% | 0.71 | 0.72 | 0.71 |
| **LogReg + TF-IDF** | 77.3% | 0.76 | 0.77 | 0.76 |
| **Standard LSTM** | 80.0% | 0.81 | 0.80 | 0.80 |
| **Tribrid Model (Ours)** | **88.1%** | **0.88** | **0.88** | **0.88** |

> **Key Takeaway:** The Tribrid model outperformed the Machine Learning baseline by **~11%** by explicitly modeling the structure of the abstract, resolving the confusion between *Background* and *Objective*.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* TensorFlow 2.x
* Scikit-Learn
* Pandas / NumPy
* NLTK (for stopwords/lemmatization)

### Running the Notebook
1. **Clone the repository:**
```bash
   git clone https://github.com/geosh0/pubmed-rct-classifier.git
```

## 🚀 Future Work
To push accuracy beyond 88%, future iterations will explore:
* **BioBERT** / **PubMedBERT**: Replacing the custom LSTM embeddings with Transformers pre-trained specifically on biomedical corpora.
* **Character-level Embeddings**: Adding a fourth input stream to handle out-of-vocabulary medical terminology.
