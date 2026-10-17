"""
Part 4: Visualization
Visualizes embeddings using:
- PCA
- t-SNE
Datasets: arXiv and AG News.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import get_logger

log = get_logger(__name__)

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def visualize(ds_name, method, dim=100, n_words=200):
    print(f"Visualizing {ds_name} | {method} (dim={dim})...")
    
    try:
        if method in ["CBOW", "Skip-gram", "FastText"]:
            if method == "FastText":
                model = FastText.load(str(MODELS_DIR / f"{ds_name}_fasttext_{dim}.model"))
            else:
                model = Word2Vec.load(str(MODELS_DIR / f"{ds_name}_{method.lower().replace('-', '')}_{dim}.model"))
            
            words = list(model.wv.index_to_key[:n_words])
            vectors = np.array([model.wv[w] for w in words])
                
        elif method == "GloVe":
            path = MODELS_DIR / f"{ds_name}_glove_{dim}.pkl"
            if not path.exists():
                print(f"  Skipping GloVe: model file not found.")
                return
            with open(path, "rb") as f:
                glove_dict = pickle.load(f)
            
            words = list(glove_dict.keys())[:n_words]
            vectors = np.array([glove_dict[w] for w in words])
    except Exception:
        print(f"  Skipping {method}: error loading model.")
        return

    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, edgecolors='k')
    for i, word in enumerate(words):
        if i % 5 == 0: 
            plt.annotate(word, xy=(pca_result[i, 0], pca_result[i, 1]))
    plt.title(f"PCA Visualization: {ds_name} | {method} (Dim={dim})")
    plt.savefig(PLOTS_DIR / f"pca_{ds_name}_{method.lower().replace('-', '')}_{dim}.png")
    plt.close()
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, n_words-1), max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5, edgecolors='k')
    for i, word in enumerate(words):
        if i % 5 == 0:
            plt.annotate(word, xy=(tsne_result[i, 0], tsne_result[i, 1]))
    plt.title(f"t-SNE Visualization: {ds_name} | {method} (Dim={dim})")
    plt.savefig(PLOTS_DIR / f"tsne_{ds_name}_{method.lower().replace('-', '')}_{dim}.png")
    plt.close()

def main():
    datasets = ["arxiv", "ag_news"]
    methods = ["CBOW", "Skip-gram", "FastText", "GloVe"]
    for ds in datasets:
        for m in methods:
            visualize(ds, m, dim=100)
    print(f"Done. Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
