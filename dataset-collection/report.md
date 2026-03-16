⚡ vrajpat3ll@orion (dataset-collection) dataset-collection> uv run .\src\collect.py --limit 5000

Global Dataset Summary (Lab 1 - Task 1)
=====================================================
+-------------------------+-------------------------------------+
| Property                | Value                               |
+-------------------------+-------------------------------------+
| Number of Documents     | 20,000                              |
| Number of Sentences     | 152,434                             |
| Average Sentence Length | 27.88                               |
| Total Tokens            | 4,249,323                           |
| Source                  | arXiv API via `arxiv` Python client |
+-------------------------+-------------------------------------+

Per-Label Summary
======================================================================
              Number of Documents Number of Sentences  Average Sentence Length Total Tokens
Label
Algorithms                  5,000              38,574                    31.41    1,211,438
NLP/AI                      5,000              37,207                    26.80      997,258
Systems/Infra               5,000              37,577                    27.03    1,015,828
Vision                      5,000              39,076                    26.23    1,024,799


⚡ vrajpat3ll@orion (dataset-collection) dataset-collection> uv run .\src\run_analysis.py --data .\data\processed\arxiv_5000_cleaned.csv
════════════════════════════════════════════════════════════
                        LOADING DATA
════════════════════════════════════════════════════════════
  Loaded 20,000 abstracts from data\processed\arxiv_5000_cleaned.csv
════════════════════════════════════════════════════════════
                       PREPROCESSING
════════════════════════════════════════════════════════════
  Total tokens      : 2,450,954
  Total sentences   : 151,881
════════════════════════════════════════════════════════════
                TASK 4 — VOCABULARY ANALYSIS
════════════════════════════════════════════════════════════
────────────────────────────────────────────────────
               VOCABULARY STATISTICS
────────────────────────────────────────────────────
  Total tokens (with repetition) : 2,450,954
  Vocabulary size (unique types)  : 49,971
  Hapax legomena                  : 16,703  (33.43% of vocab)
  Average token frequency         : 49.0475
  Most frequent token count       : 15,156

  Top 50 tokens:
  Rank   Token                      Frequency
  ────   ───────────────────────   ─────────
  1      models                        15,156
  2      model                         13,594
  3      based                         13,358
  4      data                          12,223
  5      time                          11,678
  6      performance                   10,068
  7      algorithm                      9,433
  8      problem                        9,001
  9      framework                      8,927
  10     network                        8,918
  11     language                       8,152
  12     results                        7,706
  13     learning                       7,696
  14     across                         7,579
  15     large                          7,391
  16     show                           7,311
  17     using                          7,180
  18     propose                        7,140
  19     training                       6,499
  20     high                           6,415
  21     reasoning                      6,398
  22     paper                          6,291
  23     methods                        6,234
  24     multi                          6,102
  25     two                            5,906
  26     existing                       5,865
  27     demonstrate                    5,598
  28     however                        5,582
  29     graph                          5,547
  30     real                           5,466
  31     algorithms                     5,422
  32     approach                       5,351
  33     introduce                      5,301
  34     networks                       5,295
  35     work                           5,291
  36     state                          5,042
  37     generation                     5,040
  38     first                          5,031
  39     also                           4,806
  40     llms                           4,772
  41     information                    4,756
  42     tasks                          4,689
  43     method                         4,685
  44     accuracy                       4,551
  45     experiments                    4,525
  46     task                           4,340
  47     quality                        4,318
  48     level                          4,284
  49     image                          4,214
  50     new                            4,190
────────────────────────────────────────────────────
  [zipf_plot] saved -> plots\zipf_plot.png
  [freq_histogram] saved -> plots\freq_histogram.png
════════════════════════════════════════════════════════════
                  TASK 5 — N-GRAM ANALYSIS
════════════════════════════════════════════════════════════
────────────────────────────────────────────────────────────
                      Top 20 UNIGRAMS
               Total unique unigrams: 49,971
────────────────────────────────────────────────────────────
  Rank   N-gram                                     Freq
  ────   ────────────────────────────────────   ───────
  1      models                                   15,156
  2      model                                    13,594
  3      based                                    13,358
  4      data                                     12,223
  5      time                                     11,678
  6      performance                              10,068
  7      algorithm                                 9,433
  8      problem                                   9,001
  9      framework                                 8,927
  10     network                                   8,918
  11     language                                  8,152
  12     results                                   7,706
  13     learning                                  7,696
  14     across                                    7,579
  15     large                                     7,391
  16     show                                      7,311
  17     using                                     7,180
  18     propose                                   7,140
  19     training                                  6,499
  20     high                                      6,415

────────────────────────────────────────────────────────────
                       Top 20 BIGRAMS
              Total unique bigrams: 1,180,585
────────────────────────────────────────────────────────────
  Rank   N-gram                                     Freq
  ────   ────────────────────────────────────   ───────
  1      language models                           4,278
  2      state art                                 3,489
  3      large language                            3,441
  4      real world                                2,664
  5      models llms                               1,968
  6      extensive experiments                     1,733
  7      real time                                 1,677
  8      large scale                               1,599
  9      fine tuning                               1,495
  10     reinforcement learning                    1,367
  11     results show                              1,294
  12     fine grained                              1,293
  13     results demonstrate                       1,193
  14     vision language                           1,095
  15     polynomial time                           1,065
  16     experimental results                      1,047
  17     end end                                   1,046
  18     et al                                       985
  19     machine learning                            961
  20     high quality                                953

────────────────────────────────────────────────────────────
                      Top 20 TRIGRAMS
              Total unique trigrams: 1,898,026
────────────────────────────────────────────────────────────
  Rank   N-gram                                     Freq
  ────   ────────────────────────────────────   ───────
  1      large language models                     2,973
  2      language models llms                      1,963
  3      vision language models                      591
  4      state art performance                       536
  5      achieves state art                          533
  6      large language model                        458
  7      extensive experiments demonstrate           436
  8      experimental results demonstrate            384
  9      outperforms state art                       377
  10     state art methods                           377
  11     multimodal large language                   350
  12     language models vlms                        345
  13     retrieval augmented generation              325
  14     internet things iot                         305
  15     reinforcement learning rl                   303
  16     language model llm                          300
  17     experimental results show                   295
  18     language models mllms                       281
  19     supervised fine tuning                      271
  20     deep reinforcement learning                 259

────────────────────────────────────────────────────────────
                        OBSERVATIONS
────────────────────────────────────────────────────────────
  Unique unigrams  : 49,971
  Unique bigrams   : 1,180,585  (51.4% of all bigram tokens are unique)
  Unique trigrams  : 1,898,026  (88.4% of all trigram tokens are unique)

  Most common bigram  : 'language models'  (freq=4,278)
  Most common trigram : 'large language models'  (freq=2,973)

  Hapax bigrams    : 911,653  (77.2% of unique bigrams)
  Hapax trigrams   : 1,794,872  (94.6% of unique trigrams)
  -> High hapax % indicates data sparsity; expect smoothing to matter in Lab 3.
────────────────────────────────────────────────────────────
════════════════════════════════════════════════════════════
Done. Plots saved to: plots
════════════════════════════════════════════════════════════


⚡ vrajpat3ll@orion (dataset-collection) dataset-collection> uv run src/run_split.py --data data/processed/arxiv_5000_cleaned.csv --outdir data/splits/
══════════════════════════════════════════════════════════════
                         LOADING DATA
══════════════════════════════════════════════════════════════
  Loaded 20,000 rows from data\processed\arxiv_5000_cleaned.csv
  Columns : ['arxiv_id', 'title', 'abstract', 'abstract_cleaned', 'primary_cat', 'all_cats', 'label']
══════════════════════════════════════════════════════════════
                          SPLITTING
══════════════════════════════════════════════════════════════
  Seed     : 42
  Strategy : stratified on 'label'
  Ratios   : train=80%  val=10%  test=10%

──────────────────────────────────────────────────────────────
                     DATASET SPLIT REPORT
──────────────────────────────────────────────────────────────
  Split         Count    % of total
  ────────   ───────   ───────────
  train        16,000         80.0%
  val           2,000         10.0%
  test          2,000         10.0%
  ────────   ───────   ───────────
  TOTAL        20,000        100.0%

  Class distribution (stratified on 'label'):

  Class                     Train      Val     Test    Train%
  ──────────────────────   ──────   ──────   ──────   ───────
  Algorithms                4,000      500      500     25.0%
  NLP/AI                    4,000      500      500     25.0%
  Systems/Infra             4,000      500      500     25.0%
  Vision                    4,000      500      500     25.0%
──────────────────────────────────────────────────────────────
  Random seed : 42
  Strategy    : Stratified split on 'label'
──────────────────────────────────────────────────────────────
══════════════════════════════════════════════════════════════
                            SAVING
══════════════════════════════════════════════════════════════
  [train] saved -> data\splits\train.csv  (16,000 rows)
  [  val] saved -> data\splits\val.csv  (2,000 rows)
  [ test] saved -> data\splits\test.csv  (2,000 rows)
══════════════════════════════════════════════════════════════
Done.
══════════════════════════════════════════════════════════════