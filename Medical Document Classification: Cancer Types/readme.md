# 🏥 Medical Document Classification: Cancer Types
![alt text](https://img.shields.io/badge/Python-3.8%2B-blue)

![alt text](https://img.shields.io/badge/TensorFlow-2.x-orange)

![alt text](https://img.shields.io/badge/sklearn-Latest-green)

![alt text](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Project Overview
This project implements a complete **Natural Language Processing (NLP)** pipeline to classify medical research papers into specific cancer categories (**Thyroid, Colon, and Lung Cancer**).

Unlike standard "clean" datasets, this project focuses heavily on **Real-World Data Cleaning**. The raw text contained significant PDF parsing artifacts, encoding errors, and academic boilerplate, requiring a robust, custom preprocessing pipeline before modeling.

The solution compares **Classical Machine Learning** (SVM, Random Forest) against **Deep Learning** architectures (LSTM, CNN) to determine the most effective approach for biomedical text classification.

## 📊 The Dataset
* **Source**: [Biomedical Text Publication Classification (via Kaggle)](https://www.kaggle.com/datasets/falgunipatel19/biomedical-text-publication-classification/code)
* **Content**: Full-text excerpts from medical research papers.
* **Classes**:
  * Thyroid_Cancer
  * Colon_Cancer
  * Lung_Cancer
* **Challenges**:
  * High Noise: Text contained encoding artifacts (e.g., ï¬, Ã), ligatures, and PDF metadata.
  * Boilerplate: Recurring academic phrases ("Creative Commons", "Background", "Methods") that do not contribute to classification.

## 🛠️ Methodology
### 1. Data Engineering & Preprocessing (src/medical_utils.py)
Instead of a "spaghetti code" approach, a reusable TextPreprocessor class was built to handle:
* Artifact Mapping: Automated replacement of corrupt Unicode characters (e.g., mapping ï¬ to fi).
* Regex Cleaning: Removal of non-ASCII characters and academic boilerplate using compiled Regex patterns.
* Normalization: Lemmatization and custom stop-word removal (filtering out domain-specific fillers like "et al", "study", "patient").
### 2. Exploratory Data Analysis (EDA)
* N-Gram Analysis: Visualizing the most common Bigrams (2-word phrases) per cancer type.
* Length Distribution: Analyzing token counts to detect outliers or data leakage.
* Word Clouds: Generating class-specific visualizations to verify the cleaning pipeline.
### 3. Model Architecture
Two distinct approaches were benchmarked:
1. Classical Machine Learning (Scikit-Learn)
  * Features: TF-IDF Vectorization (n-grams=1,2, max_features=7000).
  * **Models**:
  * Multinomial Naive Bayes
    * Logistic Regression (Class Weighted)
    * Linear SVM
    * Random Forest
2. Deep Learning (TensorFlow/Keras)
* Features: Learnable Word Embeddings (Trainable).
* **Architectures:**
  * LSTM: Long Short-Term Memory network to capture sequential context.
  * CNN (1D): Convolutional Neural Network with Global Max Pooling to detect specific medical keywords regardless of position.

 
