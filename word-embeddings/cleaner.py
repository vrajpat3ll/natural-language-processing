"""
cleaner.py

Pure-function text cleaning pipeline for arXiv CS abstracts.
Each function takes a raw string and returns a cleaned string.
Compose them via `clean()` for the full pipeline.

Intentionally lighter than the full normalization pipeline
(no stemming/lemmatization here).
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# One-time NLTK resource check
def _ensure_nltk_resources() -> None:
    """Download required NLTK data if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_resources()

_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))


def strip_latex(text: str) -> str:
    """
    Remove LaTeX markup common in arXiv abstracts.

    Handles:
      - Display math:  $$...$$
      - Inline math:   $...$
      - Commands:      \\command{...}  or  \\command
      - Environments:  \\begin{...} ... \\end{...}
      - Curly braces left over after command removal
    """
    # Display math first (must come before inline math)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    # Inline math
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    # \begin{env} ... \end{env}
    text = re.sub(r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", " ", text, flags=re.DOTALL)
    # \command{content}  ->  keep content (the actual words matter)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # Bare commands like \alpha, \mathbb, \rightarrow
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    # Leftover braces / brackets
    text = re.sub(r"[{}\[\]]", " ", text)
    return text


def strip_urls(text: str) -> str:
    """Remove http/https/ftp URLs and bare www. addresses."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ftp://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    return text


def strip_citations(text: str) -> str:
    """
    Remove common inline citation patterns found in abstracts.
    e.g.  [1], [12], [1,2,3], (Smith et al., 2020)
    """
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)
    text = re.sub(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)", " ", text)
    return text


def to_lowercase(text: str) -> str:
    return text.lower()


def strip_punctuation(text: str) -> str:
    """
    Remove punctuation characters, replacing with a space so adjacent
    tokens don't accidentally merge  (e.g. 'end.Start' -> 'end Start').
    Hyphens inside words are kept as spaces so 'state-of-the-art' tokenises
    as three words rather than one unknown token.
    """
    # Hyphens between word characters -> space
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    # Everything else in string.punctuation -> space
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(table)


def strip_digits(text: str) -> str:
    """
    Remove standalone digit tokens (e.g. '2', '42').
    Keeps alphanumeric tokens like 'gpt4' or 'resnet50' intact.
    """
    return re.sub(r"\b\d+\b", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces / tabs / newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    """
    Remove English stopwords using NLTK's list.
    Input is expected to already be lowercased.
    Returns a space-joined string (not a list) for consistency.
    """
    tokens = word_tokenize(text)
    filtered = [tok for tok in tokens if tok not in _STOPWORDS]
    return " ".join(filtered)


def tokenize(text: str) -> list[str]:
    """
    Final tokenization step.
    Returns a list of word tokens, stripping any residual
    single-character tokens (punctuation stragglers).
    """
    tokens = word_tokenize(text)
    return [tok for tok in tokens if len(tok) > 1]


# Composed pipelines
def clean(text: str, *, remove_stops: bool = True) -> str:
    """
    Full cleaning pipeline -> returns a cleaned string (not tokenized).

    Steps (in order):
      1. strip_latex        — arXiv-specific, must be first
      2. strip_urls
      3. strip_citations
      4. to_lowercase
      5. strip_punctuation
      6. strip_digits
      7. normalize_whitespace
      8. remove_stopwords   (optional, on by default)

    Args:
        text:           Raw abstract string.
        remove_stops:   Set False to keep stopwords
                        (e.g. for POS tagging, where 'the', 'a' matter).

    Returns:
        Cleaned string, whitespace-normalised.
    """
    text = strip_latex(text)
    text = strip_urls(text)
    text = strip_citations(text)
    text = to_lowercase(text)
    text = strip_punctuation(text)
    text = strip_digits(text)
    text = normalize_whitespace(text)
    if remove_stops:
        text = remove_stopwords(text)
        text = normalize_whitespace(text)  # word_tokenize can add spaces
    return text


def clean_and_tokenize(text: str, *, remove_stops: bool = True) -> list[str]:
    """
    Convenience wrapper: clean() followed by tokenize().
    Returns a list of tokens.
    """
    return tokenize(clean(text, remove_stops=remove_stops))

if __name__ == '__main__':
    prompt = """We study driver's optimal trajectory planning under uncertainty in the duration of a traffic light's green phase. We interpret this as an optimal control problem with an objective of minimizing the expected cost based on the fuel use, discomfort from rapid velocity changes, and time to destination. Treating this in the framework of dynamic programming, we show that the probability distribution on green phase durations gives rise to a sequence of Hamilton-Jacobi-Bellman PDEs, which are then solved numerically to obtain optimal acceleration/braking policy in feedback form. Our numerical examples illustrate the approach and highlight the role of conflicting goals and uncertainty in shaping drivers' behavior."""
    
    print(clean_and_tokenize(prompt))