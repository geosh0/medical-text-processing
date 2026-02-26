# 🏥 Medical Clinical Text Processing & Classification

## 📖 Overview
This repository contains a collection of supervised Natural Language Processing (NLP) models applied to complex medical text datasets. 

The goal of this repository is to effectively process unstructured clinical data, extract meaningful features (from TF-IDF to BioBERT), and train scalable classifiers.

## 🗂️ Datasets & Tasks

This repository is split into three distinct medical NLP tasks, each housed in its own dedicated directory:

| Dataset | NLP Task | Models Applied | 
| :--- | :--- | :--- | 
| **[OHSUMED](./ohsumed/)** | Multi-label Document Classification | XGBoost, SVM, BioBERT, LSTMs |
| **[Cancer Docs](./cancer_doc_classification/)** | Multi-class Document Classification | XGBoost, SVM, BioBERT, LSTMs |
| **[PubMed RCT 20k](./pubmed_rct_20k/)**| Sequential Sentence Classification | XGBoost, SVM, BioBERT, LSTMs |

## 🚀 Technical Highlights 

When dealing with large text corpora and massive label spaces, standard pandas/sklearn workflows often result in Out-Of-Memory (OOM) crashes. This project solves these bottlenecks using:

* **The Pareto Principle for Target Selection:** Applied the 80/20 rule to clinical MeSH terms to filter out ultra-rare labels, reducing the target space while maintaining 80% dataset coverage.
* **Memory-Safe Feature Extraction:** 
  * Implemented FP16 (Mixed Precision) batched inference for BioBERT embeddings via PyTorch `autocast` to halve VRAM usage.
  * Compressed massive textual feature spaces into `.npz` sparse matrices.
* **Out-of-Core Evaluation:** Wrote custom deep learning evaluation scripts that predict in memory-isolated chunks and immediately cast probability matrices to `uint8` binary labels to prevent RAM spikes.
* **Modular Pipeline:** Extracted messy regex text cleaning, boilerplate EDA, and training loops into a `src/` backend, keeping Jupyter notebooks strictly for high-level narrative and results.

## 📂 Repository Structure

```text
medical-text-processing/
│
├── README.md                  
│
├── ohsumed/                   <- 1. OHSUMED Multi-label Classification
│   ├── src/                   <- Python modules (text processing, DL utils, extractors)
│   └── ohsumed.ipynb 
│
├── cancer_doc_classification/ <- 2. Cancer Document Classification
│   ├── src/
│   └── cancer_doc_classification.ipynd
│
├── pubmed_rct_20k/            <- 3. PubMed RCT Sequence Classification
│   ├── src/ 
    └── pubmed_rct_20k.ipynd
```

## 🛠️ How to Run
Clone the repository:
```Bash
git clone https://github.com/geosh0/medical-text-processing.git
cd medical-text-processing
```
# DATA
* [OHSUMED](https://huggingface.co/datasets/community-datasets/ohsumed)
* [Medical Text Dataset -Cancer Doc Classification](https://www.kaggle.com/datasets/falgunipatel19/biomedical-text-publication-classification/code)
* [pubmed-rct20k](https://huggingface.co/datasets/armanc/pubmed-rct20k)
