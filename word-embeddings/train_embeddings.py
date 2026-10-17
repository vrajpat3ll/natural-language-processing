"""
Part 2: Training Word Embeddings
Trains CBOW, Skip-gram, FastText, and GloVe models.
Dimensions: 80, 100, 200.
Records: Method, Dim, Training Time, Vocab Size.
"""

import os
import time
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from mittens import Mittens
from tqdm import tqdm

from loader import load_arxiv, load_ag_news
from clean_text import preprocess
from utils import get_logger

log = get_logger(__name__)

# Constants
DIMENSIONS = [80, 100, 200]
GLOVE_VOCAB_SIZE = 5000  # Reduced for speed in pure Python implementation
WINDOW_SIZE = 5
MIN_COUNT = 5
WORKERS = 4

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def build_cooccurrence_matrix(sentences, word_to_idx, window_size=5):
    """Build a co-occurrence matrix for GloVe training."""
    vocab_size = len(word_to_idx)
    cooc = defaultdict(float)
    
    for sentence in tqdm(sentences, desc="Building co-occurrence matrix", leave=False):
        indices = [word_to_idx[w] for w in sentence if w in word_to_idx]
        for i, idx1 in enumerate(indices):
            # Context window
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                idx2 = indices[j]
                # GloVe weight: 1 / distance
                dist = abs(i - j)
                cooc[(idx1, idx2)] += 1.0 / dist
                
    # Use sparse matrix internally if possible, but keep dense for now as Mittens replacement
    matrix = np.zeros((vocab_size, vocab_size))
    for (i, j), val in cooc.items():
        matrix[i, j] = val
    return matrix

class GloVe:
    """A simple implementation of GloVe training using NumPy."""
    def __init__(self, n_components=100, learning_rate=0.01, x_max=100, alpha=0.75, max_iter=10):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.x_max = x_max
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X):
        """X is the co-occurrence matrix."""
        vocab_size = X.shape[0]
        
        # Initialize weights and biases
        W = (np.random.rand(vocab_size, self.n_components) - 0.5) / self.n_components
        W_tilde = (np.random.rand(vocab_size, self.n_components) - 0.5) / self.n_components
        b = (np.random.rand(vocab_size) - 0.5) / self.n_components
        b_tilde = (np.random.rand(vocab_size) - 0.5) / self.n_components

        # Non-zero indices for faster iteration
        ii, jj = X.nonzero()
        n_elements = len(ii)
        log.info(f"Training on {n_elements:,} non-zero elements.")
        
        log_X = np.log(X[ii, jj])
        weights = (np.minimum(X[ii, jj] / self.x_max, 1.0)) ** self.alpha
        
        for epoch in range(self.max_iter):
            total_cost = 0
            # Shuffle indices
            indices = np.arange(n_elements)
            np.random.shuffle(indices)
            
            pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{self.max_iter}", leave=False)
            for idx in pbar:
                i, j = ii[idx], jj[idx]
                
                # Inner product
                inner = np.dot(W[i], W_tilde[j])
                diff = inner + b[i] + b_tilde[j] - log_X[idx]
                
                # Weighted error
                cost = weights[idx] * (diff ** 2)
                total_cost += cost
                
                # Gradients
                grad = weights[idx] * diff
                
                # Update (simple SGD)
                # Store old values to avoid using updated values in the same step
                w_i_old = W[i].copy()
                
                W[i] -= self.learning_rate * grad * W_tilde[j]
                W_tilde[j] -= self.learning_rate * grad * w_i_old
                b[i] -= self.learning_rate * grad
                b_tilde[j] -= self.learning_rate * grad
                
            log.info(f"Epoch {epoch+1}/{self.max_iter}, Avg Cost: {total_cost/n_elements:.6f}")
                
        # Return combined embeddings
        return W + W_tilde

def train_glove(sentences, dim, vocab_size=GLOVE_VOCAB_SIZE):
    """Train GloVe using custom implementation."""
    # 1. Build restricted vocab
    all_tokens = [tok for sent in sentences for tok in sent]
    counts = Counter(all_tokens)
    most_common = counts.most_common(vocab_size)
    word_to_idx = {word: i for i, (word, count) in enumerate(most_common)}
    
    # 2. Build co-occurrence
    cooc_matrix = build_cooccurrence_matrix(sentences, word_to_idx, window_size=WINDOW_SIZE)
    
    # 3. Train GloVe
    log.info(f"Training GloVe (dim={dim}, vocab={vocab_size})...")
    model = GloVe(n_components=dim, max_iter=10) 
    embeddings = model.fit(cooc_matrix)
    
    # Map back to dict
    word_embeddings = {word: embeddings[word_to_idx[word]] for word in word_to_idx}
    return word_embeddings, vocab_size

def main():
    datasets = ["arxiv", "ag_news"]
    all_results = []

    for ds_name in datasets:
        log.info(f"=== Training on {ds_name} ===")
        # 1. Load and Preprocess data
        if ds_name == "arxiv":
            ds = load_arxiv()
        else:
            ds = load_ag_news()
            
        all_raw_text = ds.train['raw_text'].tolist()
        
        # Full preprocessing (sentence segmentation + cleaning)
        sentences = []
        for text in tqdm(all_raw_text, desc=f"Preprocessing {ds_name} sentences"):
            sentences.extend(preprocess(text))
        
        for dim in DIMENSIONS:
            log.info(f"--- {ds_name} | Dimension {dim} ---")
            
            # A. CBOW
            cbow_path = MODELS_DIR / f"{ds_name}_cbow_{dim}.model"
            if not cbow_path.exists():
                start = time.time()
                cbow = Word2Vec(sentences, vector_size=dim, window=WINDOW_SIZE, min_count=MIN_COUNT, sg=0, workers=WORKERS)
                duration = time.time() - start
                vocab_size = len(cbow.wv)
                all_results.append([ds_name, "CBOW", dim, f"{duration:.2f}s", vocab_size])
                cbow.save(str(cbow_path))
            else:
                log.info(f"Skipping {ds_name} CBOW {dim} (exists)")
                cbow = Word2Vec.load(str(cbow_path))
                all_results.append([ds_name, "CBOW", dim, "N/A", len(cbow.wv)])
            
            # B. Skip-gram
            sg_path = MODELS_DIR / f"{ds_name}_skipgram_{dim}.model"
            if not sg_path.exists():
                start = time.time()
                sg = Word2Vec(sentences, vector_size=dim, window=WINDOW_SIZE, min_count=MIN_COUNT, sg=1, workers=WORKERS)
                duration = time.time() - start
                vocab_size = len(sg.wv)
                all_results.append([ds_name, "Skip-gram", dim, f"{duration:.2f}s", vocab_size])
                sg.save(str(sg_path))
            else:
                log.info(f"Skipping {ds_name} Skip-gram {dim} (exists)")
                sg = Word2Vec.load(str(sg_path))
                all_results.append([ds_name, "Skip-gram", dim, "N/A", len(sg.wv)])
            
            # C. FastText
            ft_path = MODELS_DIR / f"{ds_name}_fasttext_{dim}.model"
            if dim == 200:
                 log.info(f"Skipping FastText for {ds_name} dim 200 to save disk space.")
            elif not ft_path.exists():
                start = time.time()
                # Drastically reduced bucket size to save disk space (default is 2,000,000)
                ft = FastText(sentences, vector_size=dim, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=WORKERS, bucket=50000)
                duration = time.time() - start
                vocab_size = len(ft.wv)
                all_results.append([ds_name, "FastText", dim, f"{duration:.2f}s", vocab_size])
                ft.save(str(ft_path))
            else:
                log.info(f"Skipping {ds_name} FastText {dim} (exists)")
                ft = FastText.load(str(ft_path))
                all_results.append([ds_name, "FastText", dim, "N/A", len(ft.wv)])
            
            # D. GloVe
            glove_path = MODELS_DIR / f"{ds_name}_glove_{dim}.pkl"
            if not glove_path.exists():
                start = time.time()
                glove_embeddings, g_vocab_size = train_glove(sentences, dim, GLOVE_VOCAB_SIZE)
                duration = time.time() - start
                all_results.append([ds_name, "GloVe", dim, f"{duration:.2f}s", g_vocab_size])
                with open(glove_path, "wb") as f:
                    pickle.dump(glove_embeddings, f)
            else:
                log.info(f"Skipping {ds_name} GloVe {dim} (exists)")
                with open(glove_path, "rb") as f:
                    glove_embeddings = pickle.load(f)
                all_results.append([ds_name, "GloVe", dim, "N/A", len(glove_embeddings)])

    # Report
    df_results = pd.DataFrame(all_results, columns=["Dataset", "Method", "Dim", "Training Time", "Vocab Size"])
    print("\n=== Part 2: Training Report ===")
    print(df_results.to_string(index=False))
    df_results.to_csv("training_report.csv", index=False)

if __name__ == "__main__":
    main()
