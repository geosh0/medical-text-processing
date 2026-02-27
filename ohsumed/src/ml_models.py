import time
import gc
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier  
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score

def get_classifiers():
    return {
        "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='liblinear'), n_jobs=1),
        "Linear SVM": OneVsRestClassifier(LinearSVC(dual=False, max_iter=2000), n_jobs=1),
        "Multinomial Naive Bayes": OneVsRestClassifier(MultinomialNB(), n_jobs=1),    
        "XGBoost": OneVsRestClassifier(XGBClassifier(n_jobs=-1, tree_method='hist', eval_metric='logloss'), n_jobs=1)
    }

def run_classical_ml_experiment(feature_sets_train, feature_sets_test, y_train, y_test, results_csv):
    column_order =[
        "Feature Set", "Classifier", "Train Time (s)", "Accuracy (Exact Match)", 
        "Hamming Loss", "Precision (Micro)", "Precision (Macro)", 
        "Recall (Micro)", "Recall (Macro)", "F1 (Micro)", "F1 (Macro)"
    ]
    
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        for col in column_order:
            if col not in results_df.columns: results_df[col] = np.nan
    else:
        results_df = pd.DataFrame(columns=column_order)

    classifiers = get_classifiers()
    
    for feat_name, X_train_raw in feature_sets_train.items():
        X_test_raw = feature_sets_test[feat_name]
        is_sparse = sp.issparse(X_train_raw)
        
        # Scaling
        scaler = MaxAbsScaler() if is_sparse else StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        for clf_name, clf in classifiers.items():
            # Professional Skipping Logic
            if clf_name == "Multinomial Naive Bayes" and not is_sparse: continue
            if clf_name == "XGBoost" and "TF-IDF" in feat_name: continue
            if clf_name in ["Logistic Regression", "Linear SVM", "XGBoost"] and "BioBERT" in feat_name: continue
            if not results_df.empty and not results_df[(results_df['Feature Set'] == feat_name) & (results_df['Classifier'] == clf_name)].empty:
                continue
                
            print(f"Training {clf_name} on {feat_name}...")
            start_time = time.time()
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            train_time = time.time() - start_time
            
            new_result = {
                "Feature Set": feat_name, "Classifier": clf_name, "Train Time (s)": round(train_time, 2),
                "Accuracy (Exact Match)": round(accuracy_score(y_test, y_pred), 4),
                "Hamming Loss": round(hamming_loss(y_test, y_pred), 4),
                "Precision (Micro)": round(precision_score(y_test, y_pred, average='micro', zero_division=0), 4),
                "Precision (Macro)": round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
                "Recall (Micro)": round(recall_score(y_test, y_pred, average='micro', zero_division=0), 4),
                "Recall (Macro)": round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
                "F1 (Micro)": round(f1_score(y_test, y_pred, average='micro', zero_division=0), 4),
                "F1 (Macro)": round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4)
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)
            results_df[column_order].to_csv(results_csv, index=False)
            gc.collect()

    return results_df.sort_values(by="F1 (Micro)", ascending=False)