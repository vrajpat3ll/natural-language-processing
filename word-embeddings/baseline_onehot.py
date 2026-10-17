"""
Part 1: Baseline Representation
Implements One-hot encoding and discusses its limitations:
- Dimensionality
- Sparsity
- Lack of semantic similarity
"""

import numpy as np
import pandas as pd
from loader import load_arxiv
from vocab import build_vocab, flatten_tokens, vocab_stats

def main():
    print("=== Part 1: Baseline Representation (One-hot Encoding) ===")
    
    # 1. Load data
    try:
        arxiv = load_arxiv()
    except FileNotFoundError:
        print("Error: Data files not found. Please ensure data/splits/ exists.")
        return

    train_text = arxiv.train['text'].tolist()
    
    # 2. Build vocabulary
    # The 'text' column is already cleaned and space-separated
    tokenized_train = [text.split() for text in train_text]
    flat_tokens = flatten_tokens(tokenized_train)
    vocab_counter = build_vocab(flat_tokens)
    stats = vocab_stats(vocab_counter)
    
    full_vocab_size = stats['vocab_size']
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Full Vocabulary Size: {full_vocab_size:,}")
    
    # For demonstration, we'll use a limited vocabulary to keep it readable
    demo_vocab_size = 5000
    most_common = vocab_counter.most_common(demo_vocab_size)
    word_to_idx = {word: i for i, (word, count) in enumerate(most_common)}
    
    # 3. One-hot encoding demonstration
    def get_one_hot(word, size):
        vec = np.zeros(size)
        if word in word_to_idx:
            vec[word_to_idx[word]] = 1
        return vec

    test_words = ["diffusion", "models", "learning", "network", "image"]
    print(f"\nExample One-hot vectors (Dimension: {demo_vocab_size}):")
    for word in test_words:
        vec = get_one_hot(word, demo_vocab_size)
        # Show a slice where the '1' might be if it's in top 20
        idx = word_to_idx.get(word, -1)
        if 0 <= idx < 20:
            print(f"  {word:10} [index {idx:3}]: {vec[:20]} ...")
        else:
            print(f"  {word:10} [index {idx:3}]: {vec[:10]} ... [1 at {idx}] ...")

    # 4. Discussion
    print("\n--- Discussion ---")
    
    # Dimensionality
    print("1. Dimensionality Issues:")
    print(f"   - Full vocabulary size is {full_vocab_size:,}. Each word requires a vector of this length.")
    print("   - Storing a matrix for 5,000 documents with this dimension would take approx.")
    print(f"     {5000 * full_vocab_size * 8 / (1024**2):.2f} MB (if dense float64).")
    print("   - Scaling to millions of words is computationally prohibitive.")

    # Sparsity
    sparsity = (full_vocab_size - 1) / full_vocab_size
    print("\n2. Sparsity:")
    print(f"   - Each vector has exactly one '1' and {full_vocab_size-1:,} '0's.")
    print(f"   - Sparsity level: {sparsity:.6%}")
    print("   - Most of the memory is wasted on zeros, requiring sparse matrix formats for efficiency.")

    # Lack of semantic similarity
    w1, w2 = "diffusion", "images"
    v1 = get_one_hot(w1, demo_vocab_size)
    v2 = get_one_hot(w2, demo_vocab_size)
    similarity = np.dot(v1, v2)
    
    print("\n3. Lack of Semantic Similarity:")
    print(f"   - Dot product between '{w1}' and '{w2}': {similarity}")
    print("   - Since all one-hot vectors are orthogonal (dot product = 0 for any i != j),")
    print(f"     they fail to capture that '{w1}' and '{w2}' might be semantically related in our corpus.")
    print("   - Every word is equally 'distant' from every other word.")

if __name__ == "__main__":
    main()
