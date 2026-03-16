"""
ArXiv CS Papers — NLP Lab Data Analyzer
Run  : marimo run analyzer.py
Deps : uv add marimo datasets pandas plotly nltk numpy

Marimo contract rules followed here:
  • Every public variable is defined (returned) in exactly ONE cell.
  • All dependencies enter a cell via its argument list — no implicit globals.
  • Private (_-prefixed) variables are scoped to the cell that creates them.
  • nltk downloads happen once in _nltk_setup; callables flow via args.
  • `mo` originates from _mo_cell and flows to all other cells via args.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _header(mo):
    mo.md("""
    <div style="
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 1rem;
        border: 1px solid #2a2a4a;
        font-family: 'JetBrains Mono', monospace;
    ">
        <div style="font-size:0.75rem;color:#6c63ff;letter-spacing:0.3em;
                    text-transform:uppercase;margin-bottom:0.5rem;">
            IIT Jammu · NLP Lab · CS-4-29
        </div>
        <h1 style="font-size:2.4rem;font-weight:700;margin:0 0 0.4rem 0;
                   background:linear-gradient(90deg,#e0e0ff,#a78bfa,#60a5fa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ArXiv CS Papers — Corpus Analyzer
        </h1>
        <p style="color:#94a3b8;margin:0;font-size:0.95rem;">
            Interactive analysis of
            <a href="https://huggingface.co/datasets/vrajpat3ll/arxiv-cs-papers">
            <code style="color:#a78bfa;">vrajpat3ll/arxiv-cs-papers</code>
            </a> · 20,000 abstracts · 4 CS categories
        </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _imports():
    import re
    import math
    import warnings
    from collections import Counter

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import marimo as mo
    import nltk

    warnings.filterwarnings("ignore")

    mo.md("""# Import Dependencies""")
    return Counter, math, mo, nltk, np, pd, px, re


@app.cell(hide_code=True)
def _nltk_setup(mo, nltk):
    for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords as _sw_corpus
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    try:
        english_stopwords = set(_sw_corpus.words("english"))
    except Exception:
        english_stopwords = set()

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    mo.md("""## NLTK Setup""")
    return english_stopwords, lemmatizer, sent_tokenize, stemmer, word_tokenize


@app.cell(hide_code=True)
def _load_controls(mo):
    split_selector = mo.ui.dropdown(
        options=["train", "validation", "test", "all"],
        value="all",
        label="Dataset Split",
    )
    sample_slider = mo.ui.slider(
        start=500,
        stop=20000,
        step=500,
        value=5000,
        label="Max rows to load",
    )
    mo.md(f"""
    ### ⚙️ Load Settings
    {mo.hstack([split_selector, sample_slider], gap=2)}
    """)
    return sample_slider, split_selector


@app.cell(hide_code=True)
def _load_data(mo, np, pd, sample_slider, split_selector):
    _spin = mo.status.spinner(title="Loading dataset from HuggingFace…")
    _load_ok = False
    with _spin:
        try:
            from datasets import load_dataset as _hf

            _split = split_selector.value
            if _split == "all":
                _parts = [
                    _hf("vrajpat3ll/arxiv-cs-papers", split=_s)
                    .to_pandas()
                    .assign(split=_s)
                    for _s in ("train", "validation", "test")
                ]
                df = pd.concat(_parts, ignore_index=True)
            else:
                df = (
                    _hf("vrajpat3ll/arxiv-cs-papers", split=_split)
                    .to_pandas()
                    .assign(split=_split)
                )
            if len(df) > sample_slider.value:
                df = df.sample(n=sample_slider.value, random_state=42).reset_index(
                    drop=True
                )
            _load_ok = True
            _load_err = ""
        except Exception as _e:
            _rng = np.random.default_rng(42)
            _labels = ["Algorithms", "NLP/AI", "Systems/Infra", "Vision"]
            _n = min(sample_slider.value, 1000)
            df = pd.DataFrame(
                {
                    "abstract": [
                        f"This paper presents a novel approach to "
                        f"{_rng.choice(['optimization', 'inference', 'training', 'classification'])} "
                        f"using {_rng.choice(['transformers', 'graphs', 'kernels', 'attention'])} methods."
                        for _ in range(_n)
                    ],
                    "label": _rng.choice(_labels, size=_n),
                    "split": _rng.choice(
                        ["train", "validation", "test"], size=_n, p=[0.8, 0.1, 0.1]
                    ),
                    "arxiv_id": [f"2301.{i:05d}" for i in range(_n)],
                }
            )
            _load_ok = False
            _load_err = str(_e)

    if _load_ok:
        mo.callout(
            mo.md(f"✅ **Loaded {len(df):,} rows** from HuggingFace."), kind="success"
        )
    else:
        mo.callout(
            mo.md(
                f"⚠️ **HuggingFace unavailable** — synthetic demo data in use.\n\n`{_load_err}`"
            ),
            kind="warn",
        )
    return (df,)


@app.cell
def _tokenise_corpus(df, sent_tokenize, word_tokenize):
    _texts = df["abstract"].dropna().tolist()
    all_sents = []
    word_tokens = []
    for _t in _texts:
        all_sents.extend(sent_tokenize(_t))
        word_tokens.extend(w for w in word_tokenize(_t.lower()) if w.isalpha())
    return all_sents, word_tokens


@app.cell(hide_code=True)
def _corpus_stats(all_sents, df, mo, word_tokens):
    _stats = {
        "Documents": f"{len(df):,}",
        "Sentences": f"{len(all_sents):,}",
        "Avg Sents/Doc": f"{len(all_sents) / max(len(df), 1):.1f}",
        "Total Tokens": f"{len(word_tokens):,}",
        "Vocabulary Size": f"{len(set(word_tokens)):,}",
        "Source": "HuggingFace · vrajpat3ll/arxiv-cs-papers",
    }
    _cards = "".join(
        f"""<div style="background:#1e1e3a;border-radius:12px;padding:1.1rem 1.4rem;
                        border:1px solid #2a2a4a;min-width:150px;flex:1;">
                <div style="color:#6c63ff;font-size:0.68rem;letter-spacing:0.12em;
                            text-transform:uppercase;margin-bottom:0.25rem;">{k}</div>
                <div style="color:#e2e8f0;font-size:1.05rem;font-weight:600;
                            font-family:'JetBrains Mono',monospace;">{v}</div>
            </div>"""
        for k, v in _stats.items()
    )
    mo.md(f"""
    ### 📊 Corpus Statistics
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.5rem;">{_cards}</div>
    """)
    return


@app.cell
def _per_label_stats(df, mo, pd, word_tokenize):
    _rows = []
    for _lbl, _grp in df.groupby("label"):
        _toks = [
            w
            for _t in _grp["abstract"].dropna()
            for w in word_tokenize(_t.lower())
            if w.isalpha()
        ]
        _rows.append(
            {
                "Label": _lbl,
                "Documents": len(_grp),
                "Tokens": len(_toks),
                "Vocab Size": len(set(_toks)),
                "Avg Tokens/Doc": round(len(_toks) / max(len(_grp), 1), 1),
            }
        )
    label_df = pd.DataFrame(_rows)
    mo.vstack(
        [
            mo.md("### 🏷️ Per-Label Breakdown"),
            mo.ui.table(label_df, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _class_imbalance(df, mo, pd, px):
    _cnt = df["label"].value_counts().reset_index()
    _cnt.columns = ["Label", "Count"]
    _total = _cnt["Count"].sum()

    _cnt["Share (%)"] = (_cnt["Count"] / _total * 100).round(2)
    _cnt["Ratio to Max"] = (_cnt["Count"] / _cnt["Count"].max()).round(3)

    _max_c = _cnt["Count"].max()
    _min_c = _cnt["Count"].min()
    _ir = round(_max_c / _min_c, 3)
    _ideal = _total / len(_cnt)
    _chi2 = round(sum((_cnt["Count"] - _ideal) ** 2 / _ideal), 2)

    if _ir < 1.5:
        _sev, _sev_kind = "Balanced ✅", "success"
        _advice = (
            "**No action needed.** Classes are well-balanced — "
            "standard cross-entropy and accuracy metrics are appropriate."
        )
    elif _ir < 3.0:
        _sev, _sev_kind = "Mild Imbalance ⚠️", "warn"
        _advice = (
            "**Mild imbalance detected.** Consider: "
            "(1) class-weighted loss (`class_weight='balanced'` in sklearn), "
            "(2) stratified splits, "
            "(3) macro-F1 as primary metric instead of accuracy."
        )
    else:
        _sev, _sev_kind = "Severe Imbalance 🔴", "danger"
        _advice = (
            "**Severe imbalance.** Recommended: "
            "(1) oversample minority (SMOTE / random), "
            "(2) undersample majority, "
            "(3) weighted cross-entropy, "
            "(4) report per-class precision / recall / F1."
        )

    _metrics = {
        "Imbalance Ratio": str(_ir),
        "χ² from Uniform": str(_chi2),
        "Majority Class": f"{_cnt.iloc[0]['Label']} ({_cnt.iloc[0]['Count']:,})",
        "Minority Class": f"{_cnt.iloc[-1]['Label']} ({_cnt.iloc[-1]['Count']:,})",
        "Severity": _sev,
    }
    _cards = "".join(
        f"""<div style="background:#1e1e3a;border-radius:12px;padding:1.1rem 1.4rem;
                        border:1px solid #2a2a4a;min-width:150px;flex:1;">
                <div style="color:#6c63ff;font-size:0.68rem;letter-spacing:0.12em;
                            text-transform:uppercase;margin-bottom:0.25rem;">{k}</div>
                <div style="color:#e2e8f0;font-size:1.05rem;font-weight:600;
                            font-family:'JetBrains Mono',monospace;">{v}</div>
            </div>"""
        for k, v in _metrics.items()
    )

    # actual vs ideal grouped bar
    _ideal_df = pd.DataFrame(
        {
            "Label": _cnt["Label"].tolist() * 2,
            "Count": _cnt["Count"].tolist() + [int(_ideal)] * len(_cnt),
            "Series": ["Actual"] * len(_cnt) + ["Ideal (uniform)"] * len(_cnt),
        }
    )
    _fig_bar = px.bar(
        _ideal_df,
        x="Label",
        y="Count",
        color="Series",
        barmode="group",
        color_discrete_map={"Actual": "#6c63ff", "Ideal (uniform)": "#475569"},
        template="plotly_dark",
        title="Actual vs Ideal Uniform Count per Class",
    )
    _fig_bar.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#1a1a2e",
        font_family="JetBrains Mono",
        title_font_size=15,
        legend=dict(bgcolor="#1a1a2e"),
    )

    # share pie
    _fig_pie = px.pie(
        _cnt,
        names="Label",
        values="Count",
        color_discrete_sequence=["#6c63ff", "#60a5fa", "#34d399", "#f472b6"],
        template="plotly_dark",
        title="Class Share (%)",
        hole=0.45,
    )
    _fig_pie.update_layout(
        paper_bgcolor="#0f0f1a",
        font_family="JetBrains Mono",
        title_font_size=15,
    )

    # per-split × label heatmap
    if "split" in df.columns:
        _pivot = (
            df.groupby(["split", "label"])
            .size()
            .reset_index(name="count")
            .pivot(index="split", columns="label", values="count")
            .fillna(0)
        )
        _fig_heat = px.imshow(
            _pivot,
            color_continuous_scale=["#0f0f1a", "#6c63ff", "#a78bfa"],
            template="plotly_dark",
            title="Class Count Heatmap: Split × Label",
            text_auto=True,
            aspect="auto",
        )
        _fig_heat.update_layout(
            paper_bgcolor="#0f0f1a",
            font_family="JetBrains Mono",
            title_font_size=15,
        )
        _heatmap = mo.ui.plotly(_fig_heat)
    else:
        _heatmap = mo.md("*(split column unavailable)*")

    mo.vstack(
        [
            mo.md("### ⚖️ Class Imbalance Analysis"),
            mo.md(
                f'<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;">{_cards}</div>'
            ),
            mo.callout(mo.md(_advice), kind=_sev_kind),
            mo.hstack([mo.ui.plotly(_fig_bar), mo.ui.plotly(_fig_pie)], gap=2),
            mo.md("#### Counts by Split × Label"),
            _heatmap,
            mo.md("#### Detailed Counts Table"),
            mo.ui.table(
                _cnt[["Label", "Count", "Share (%)", "Ratio to Max"]], selection=None
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _split_dist(df, mo, px):
    if "split" in df.columns:
        _sc = df["split"].value_counts().reset_index()
        _sc.columns = ["Split", "Count"]
        _fig = px.pie(
            _sc,
            names="Split",
            values="Count",
            color_discrete_sequence=["#6c63ff", "#60a5fa", "#f472b6"],
            template="plotly_dark",
            title="Train / Validation / Test Split",
            hole=0.4,
        )
        _fig.update_layout(
            paper_bgcolor="#0f0f1a",
            font_family="JetBrains Mono",
            title_font_size=16,
        )
        mo.vstack([mo.md("### 🍕 Dataset Splits"), mo.ui.plotly(_fig)])
    return


@app.cell
def _zipf_chart(Counter, math, mo, pd, px, word_tokens):
    _freq = Counter(word_tokens)
    _sf = sorted(_freq.values(), reverse=True)[:2000]
    _ranks = list(range(1, len(_sf) + 1))
    _zdf = pd.DataFrame(
        {
            "log_rank": [math.log10(r) for r in _ranks],
            "log_freq": [math.log10(f) for f in _sf],
            "zipf_expected": [math.log10(_sf[0]) - math.log10(r) for r in _ranks],
        }
    )
    _fig = px.scatter(
        _zdf,
        x="log_rank",
        y="log_freq",
        template="plotly_dark",
        labels={"log_rank": "log₁₀(Rank)", "log_freq": "log₁₀(Frequency)"},
        title="Zipf's Law — Word Frequency vs Rank",
        opacity=0.45,
    )
    _fig.add_scatter(
        x=_zdf["log_rank"],
        y=_zdf["zipf_expected"],
        mode="lines",
        name="Ideal Zipf (slope = −1)",
        line=dict(color="#f472b6", width=2, dash="dash"),
    )
    _fig.update_traces(
        marker=dict(color="#6c63ff", size=3), selector=dict(mode="markers")
    )
    _fig.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#1a1a2e",
        font_family="JetBrains Mono",
        title_font_size=16,
    )
    mo.vstack([mo.md("### 📉 Zipf's Law"), mo.ui.plotly(_fig)])
    return


@app.cell
def _ngram_controls(mo):
    ngram_type = mo.ui.dropdown(
        options={"Unigrams": 1, "Bigrams": 2, "Trigrams": 3},
        value="Unigrams",
        label="N-gram type",
    )
    topn_slider = mo.ui.slider(start=10, stop=50, step=5, value=20, label="Top N")
    mo.hstack([ngram_type, topn_slider], gap=2)
    return ngram_type, topn_slider


@app.cell
def _ngram_chart(Counter, mo, ngram_type, pd, px, topn_slider, word_tokens):
    _n = ngram_type.value
    _topn = topn_slider.value
    _grams = (
        word_tokens
        if _n == 1
        else [
            " ".join(word_tokens[i : i + _n]) for i in range(len(word_tokens) - _n + 1)
        ]
    )
    _gdf = pd.DataFrame(
        Counter(_grams).most_common(_topn), columns=["ngram", "count"]
    ).sort_values("count")
    _fig = px.bar(
        _gdf,
        x="count",
        y="ngram",
        orientation="h",
        template="plotly_dark",
        color="count",
        color_continuous_scale=["#1a1a4e", "#6c63ff", "#a78bfa"],
        title=f"Top {_topn} {ngram_type.selected_key}",
        labels={"count": "Frequency", "ngram": ""},
    )
    _fig.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#1a1a2e",
        font_family="JetBrains Mono",
        title_font_size=16,
        height=max(400, _topn * 22),
        coloraxis_showscale=False,
    )
    mo.vstack([mo.md("### 🔤 N-gram Analysis"), mo.ui.plotly(_fig)])
    return


@app.cell
def _freq_histogram(Counter, mo, px, word_tokens):
    _freq = Counter(word_tokens)
    _vals = list(_freq.values())
    _rare = sum(1 for v in _vals if v == 1)
    _fig = px.histogram(
        x=_vals,
        nbins=60,
        template="plotly_dark",
        title="Word Frequency Distribution (log y)",
        labels={"x": "Frequency", "y": "Number of Words"},
        log_y=True,
        color_discrete_sequence=["#60a5fa"],
    )
    _fig.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#1a1a2e",
        font_family="JetBrains Mono",
        title_font_size=16,
    )
    mo.vstack(
        [
            mo.md("### 📊 Word Frequency Histogram"),
            mo.ui.plotly(_fig),
            mo.callout(
                mo.md(
                    f"**Hapax Legomena** (words appearing exactly once): **{_rare:,}** "
                    f"({100 * _rare / max(len(_freq), 1):.1f}% of vocabulary)"
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _vocab_coverage(Counter, mo, pd, word_tokens):
    _freq = Counter(word_tokens)
    _sf = sorted(_freq.values(), reverse=True)
    _total = sum(_sf)
    _running = 0
    _cumul = []
    for _f in _sf:
        _running += _f
        _cumul.append(_running / _total)
    _rows = []
    for _m in [0.50, 0.80, 0.90, 0.95, 0.99]:
        for _i, _c in enumerate(_cumul):
            if _c >= _m:
                _rows.append(
                    {"Coverage": f"{int(_m * 100)}%", "Unique words needed": _i + 1}
                )
                break
    mo.vstack(
        [
            mo.md("### 📐 Vocabulary Coverage"),
            mo.ui.table(pd.DataFrame(_rows), selection=None),
            mo.md("*Minimum unique words to cover X% of all token occurrences.*"),
        ]
    )
    return


@app.cell
def _text_viewer_controls(df, mo):
    row_slider = mo.ui.slider(
        start=0,
        stop=min(len(df) - 1, 499),
        step=1,
        value=0,
        label="Row index",
    )
    mo.hstack([row_slider])
    return (row_slider,)


@app.cell
def _text_viewer(
    df,
    english_stopwords,
    lemmatizer,
    mo,
    re,
    row_slider,
    stemmer,
    word_tokenize,
):
    _idx = row_slider.value
    _row = df.iloc[_idx]
    _raw = str(_row.get("abstract", ""))

    _cleaned = re.sub(r"https?://\S+", "", _raw.lower())
    _cleaned = re.sub(r"<[^>]+>", "", _cleaned)
    _cleaned = re.sub(r"[^a-z\s]", " ", _cleaned)
    _cleaned = re.sub(r"\s+", " ", _cleaned).strip()

    _tokens = word_tokenize(_cleaned)
    _no_stop = [w for w in _tokens if w not in english_stopwords and len(w) > 1]
    _stemmed = [stemmer.stem(w) for w in _no_stop]
    _lemmatized = [lemmatizer.lemmatize(w) for w in _no_stop]

    mo.vstack(
        [
            mo.md("### 🔍 Raw vs Cleaned Text Viewer"),
            mo.md(
                f"**Label:** `{_row.get('label', '—')}` · **Split:** `{_row.get('split', '—')}`"
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**Raw Text**"),
                            mo.md(f"> {_raw[:600]}{'…' if len(_raw) > 600 else ''}"),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Cleaned -> No Stopwords**"),
                            mo.md(f"> {' '.join(_no_stop[:80])}…"),
                            mo.md("**Stemmed**"),
                            mo.md(f"> {' '.join(_stemmed[:60])}…"),
                            mo.md("**Lemmatized**"),
                            mo.md(f"> {' '.join(_lemmatized[:60])}…"),
                        ]
                    ),
                ],
                gap=2,
            ),
        ]
    )
    return


@app.cell
def _footer(mo):
    mo.md("""
    ---
    <div style="text-align:center;color:#475569;font-size:0.8rem;
                font-family:'JetBrains Mono',monospace;">
        NLP Lab · CS-4-29 · IIT Jammu · seed=42 · dataset: vrajpat3ll/arxiv-cs-papers
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
