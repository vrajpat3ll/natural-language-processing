"""
Part 5: Downstream Task
Compares BoW representation vs Embeddings for text classification.
Datasets: arXiv and AG News.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from loader import load_arxiv, load_ag_news
from utils import get_logger

log = get_logger(__name__)

MODELS_DIR = Path("models")

def get_doc_embedding(tokens, model, method, dim):
    vecs = []
    if method in ["CBOW", "Skip-gram", "FastText"]:
        for t in tokens:
            if t in model.wv:
                vecs.append(model.wv[t])
            elif method == "FastText":
                vecs.append(model.wv[t])
    elif method == "GloVe":
        for t in tokens:
            if t in model:
                vecs.append(model[t])
                
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

def evaluate_dataset(ds_name, ds):
    print(f"\n--- Dataset: {ds_name} ---")
    train_df = ds.train
    test_df = ds.test
    
    # Tokenize
    train_tokens = [t.split() for t in train_df['text']]
    test_tokens = [t.split() for t in test_df['text']]
    y_train = train_df['label']
    y_test = test_df['label']
    
    results = []
    
    # --- BoW Baseline ---
    print("Training BoW Baseline...")
    vectorizer = CountVectorizer(max_features=5000)
    X_train_bow = vectorizer.fit_transform(train_df['text'])
    X_test_bow = vectorizer.transform(test_df['text'])
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_bow, y_train)
    y_pred = clf.predict(X_test_bow)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append([ds_name, "BoW", acc, f1])
    print(f"  BoW Accuracy: {acc:.4f}")
    
    # --- Embedding Features ---
    dim = 100 # Evaluate at dim 100
    methods = ["CBOW", "Skip-gram", "FastText", "GloVe"]
    
    for method in methods:
        # Load model
        try:
            if method == "GloVe":
                path = MODELS_DIR / f"{ds_name}_glove_{dim}.pkl"
                if not path.exists(): continue
                with open(path, "rb") as f:
                    model = pickle.load(f)
            elif method == "FastText":
                model = FastText.load(str(MODELS_DIR / f"{ds_name}_fasttext_{dim}.model"))
            else:
                model = Word2Vec.load(str(MODELS_DIR / f"{ds_name}_{method.lower().replace('-', '')}_{dim}.model"))
        except Exception:
            print(f"  Skipping {method}: model file not found.")
            continue
            
        # Transform documents
        X_train_emb = np.array([get_doc_embedding(tokens, model, method, dim) for tokens in train_tokens])
        X_test_emb = np.array([get_doc_embedding(tokens, model, method, dim) for tokens in test_tokens])
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_emb, y_train)
        y_pred = clf.predict(X_test_emb)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append([ds_name, f"{method} (avg)", acc, f1])
        print(f"  {method} Accuracy: {acc:.4f}")
        
    return results

def main():
    print("=== Part 5: Downstream Task (Comparison) ===")
    
    all_results = []
    
    # 1. arXiv
    arxiv = load_arxiv()
    all_results.extend(evaluate_dataset("arxiv", arxiv))
    
    # 2. AG News
    ag_news = load_ag_news()
    all_results.extend(evaluate_dataset("ag_news", ag_news))
    
    # Final Report
    df_report = pd.DataFrame(all_results, columns=["Dataset", "Representation", "Accuracy", "F1 Score"])
    print("\n--- Downstream Task Report ---")
    print(df_report.to_string(index=False))
    df_report.to_csv("downstream_report.csv", index=False)

if __name__ == "__main__":
    main()
