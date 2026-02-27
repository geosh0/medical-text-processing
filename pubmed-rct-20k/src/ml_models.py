# ml_models.py
import time
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

def train_evaluate_ml_models(feature_sets, train_y, test_y, label_classes):
    """
    Loops through feature sets and models, training and evaluating each.
    
    feature_sets: dict -> {"Name": (X_train, X_test)}
    """
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "Linear SVM": LinearSVC(random_state=42, max_iter=2000, dual='auto')
    }

    results_list = []
    print("\n--- Starting Classical Model Benchmarking ---")

    for feat_name, (X_train, X_test) in feature_sets.items():
        for model_name, model in models.items():
            
            # Skip NB for Word2Vec (negative values)
            if model_name == "Naive Bayes" and "Word2Vec" in feat_name:
                continue

            print(f"Training {model_name} on {feat_name}...")
            start_time = time.time()
            model.fit(X_train, train_y)
            train_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(test_y, y_pred)
            report = classification_report(test_y, y_pred, target_names=label_classes, output_dict=True)
            
            results_list.append({
                "Model": model_name,
                "Features": feat_name,
                "Accuracy": acc,
                "F1-Score (Weighted)": report['weighted avg']['f1-score'],
                "Training Time (s)": round(train_time, 2)
            })
            
    return pd.DataFrame(results_list)