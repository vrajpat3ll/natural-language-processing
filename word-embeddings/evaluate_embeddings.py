"""
Part 3: Embedding Evaluation
Evaluates embeddings using:
- Cosine similarity
- Nearest neighbor search
- Word analogy tasks
"""

import pickle
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_logger

log = get_logger(__name__)

MODELS_DIR = Path("models")

def load_glove(dim):
    path = MODELS_DIR / f"glove_{dim}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def get_vector(word, model, method):
    if method in ["CBOW", "Skip-gram", "FastText"]:
        if word in model.wv:
            return model.wv[word]
        elif method == "FastText":
            return model.wv[word] # FastText handles OOV
        return None
    elif method == "GloVe":
        return model.get(word)
    return None

def find_nearest_neighbors(word, model, method, top_n=5):
    vec = get_vector(word, model, method)
    if vec is None:
        return []
    
    if method in ["CBOW", "Skip-gram", "FastText"]:
        return model.wv.most_similar(word, topn=top_n)
    
    elif method == "GloVe":
        words = list(model.keys())
        matrix = np.array([model[w] for w in words])
        sims = cosine_similarity(vec.reshape(1, -1), matrix)[0]
        indices = np.argsort(sims)[::-1][1:top_n+1]
        return [(words[i], sims[i]) for i in indices]

def analogy(w1, w2, w3, model, method, top_n=1):
    """w1 - w2 + w3 = ? (e.g., king - man + woman = queen)"""
    if method in ["CBOW", "Skip-gram", "FastText"]:
        try:
            return model.wv.most_similar(positive=[w1, w3], negative=[w2], topn=top_n)
        except KeyError:
            return []
    elif method == "GloVe":
        v1 = get_vector(w1, model, method)
        v2 = get_vector(w2, model, method)
        v3 = get_vector(w3, model, method)
        if v1 is None or v2 is None or v3 is None:
            return []
        
        target = v1 - v2 + v3
        words = list(model.keys())
        matrix = np.array([model[w] for w in words])
        sims = cosine_similarity(target.reshape(1, -1), matrix)[0]
        # Exclude input words
        sims_dict = {words[i]: sims[i] for i in range(len(words)) if words[i] not in [w1, w2, w3]}
        sorted_sims = sorted(sims_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_sims[:top_n]

def main():
    dim = 100 # Default dimension for evaluation
    
    print(f"=== Part 3: Embedding Evaluation (Dimension {dim}) ===")
    
    methods = ["CBOW", "Skip-gram", "FastText", "GloVe"]
    models = {}
    
    # Load models
    try:
        models["CBOW"] = Word2Vec.load(str(MODELS_DIR / f"cbow_{dim}.model"))
        models["Skip-gram"] = Word2Vec.load(str(MODELS_DIR / f"skipgram_{dim}.model"))
        models["FastText"] = FastText.load(str(MODELS_DIR / f"fasttext_{dim}.model"))
        models["GloVe"] = load_glove(dim)
    except Exception as e:
        print(f"Warning: Could not load some models for dim {dim}. Ensure training is complete.")
        # return # Commenting out to allow partial eval if some are missing
    
    test_words = ["diffusion", "learning", "network", "image", "quantum"]
    
    for method in methods:
        model = models.get(method)
        if model is None:
            continue
            
        print(f"\n--- Method: {method} ---")
        
        # 1. Similarity
        w_a, w_b = "learning", "training"
        v_a = get_vector(w_a, model, method)
        v_b = get_vector(w_b, model, method)
        if v_a is not None and v_b is not None:
            sim = cosine_similarity(v_a.reshape(1, -1), v_b.reshape(1, -1))[0][0]
            print(f"Similarity ('{w_a}', '{w_b}'): {sim:.4f}")
        
        # 2. Nearest Neighbors
        print("Nearest Neighbors:")
        for word in test_words[:3]:
            neighbors = find_nearest_neighbors(word, model, method)
            neighbor_str = ", ".join([f"{n} ({s:.2f})" for n, s in neighbors])
            print(f"  {word:10}: {neighbor_str}")
            
        # 3. Analogies
        # Since this is arXiv, analogies might be domain specific
        # neural - network + architecture = ?
        print("Analogy (neural - network + data):")
        res = analogy("neural", "network", "data", model, method)
        if res:
            print(f"  neural - network + data = {res[0][0]} ({res[0][1]:.2f})")

if __name__ == "__main__":
    main()
