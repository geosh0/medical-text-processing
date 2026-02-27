# feature_extraction.py
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

def get_bow_features(train_text, test_text, max_features, stop_words):
    """Generates Bag-of-Words features."""
    print("\n--- Generating Bag-of-Words Features ---")
    vectorizer = CountVectorizer(max_features=max_features, stop_words=list(stop_words))
    
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    
    print(f"BoW Train shape: {X_train.shape}")
    return X_train, X_test, vectorizer

def get_tfidf_features(train_text, test_text, max_features, stop_words):
    """Generates TF-IDF features with Bigrams."""
    print("\n--- Generating TF-IDF Features ---")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words=list(stop_words))
    
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    
    print(f"TF-IDF Train shape: {X_train.shape}")
    return X_train, X_test, vectorizer

def get_word2vec_features(train_tokens, test_tokens, vector_size=100):
    """Trains Word2Vec and averages vectors to create document embeddings."""
    print("\n--- Generating Word2Vec Features ---")
    
    # 1. Train Model
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(sentences=train_tokens, vector_size=vector_size, 
                         window=5, min_count=5, workers=4)

    # 2. Helper to average vectors
    def document_vector(tokens, model):
        feature_vec = np.zeros((vector_size,), dtype="float32")
        n_words = 0
        index2word_set = set(model.wv.index_to_key)
        
        for token in tokens:
            if token in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model.wv[token])
        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    # 3. Transform
    print("Vectorizing documents...")
    X_train = np.array([document_vector(t, w2v_model) for t in train_tokens])
    X_test = np.array([document_vector(t, w2v_model) for t in test_tokens])
    
    # 4. Scale (Standardize)
    print("Standardizing W2V features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Word2Vec Train shape: {X_train_scaled.shape}")
    return X_train_scaled, X_test_scaled