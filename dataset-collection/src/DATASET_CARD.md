---
language:
  - en
license: cc-by-4.0
task_categories:
  - text-classification
task_ids:
  - multi-class-classification
tags:
  - arxiv
  - scientific-abstracts
  - computer-science
pretty_name: ArXiv CS Abstracts
size_categories:
  - 10K - 100K
---

# ArXiv CS Abstracts

A corpus of CS paper abstracts from the [arXiv API](https://arxiv.org/help/api/index),
organised into four thematic labels.

Collected for NLP Lab (CS-5-45(MO)), IIT Jammu, Semester 2, 2025-2026.

## Label Schema

| Label         | arXiv Categories           | Theme                           |
| ------------- | -------------------------- | ------------------------------- |
| NLP/AI        | cs.CL, cs.AI, cs.LG, cs.IR | Language, reasoning, & learning |
| Vision        | cs.CV, cs.GR, cs.MM        | Visual computing & multimedia   |
| Systems/Infra | cs.NI, cs.DC, cs.OS, cs.SY | Networks, systems, & infra      |
| Algorithms    | cs.DS, cs.CC, cs.GT, cs.CG | Theory & foundations            |

Label = **primary category** (first arXiv tag). Cross-listed papers are not duplicated.

## Fields

| Column        | Description                     |
| ------------- | ------------------------------- |
| `arxiv_id`    | Unique arXiv paper ID           |
| `title`       | Paper title                     |
| `abstract`    | Raw abstract (no preprocessing) |
| `primary_cat` | Primary arXiv category          |
| `all_cats`    | All categories (pipe-separated) |
| `label`       | One of the four thematic labels |

## Collection details

- Source: [arxiv.py](https://github.com/lukasschwab/arxiv.py)
- Sort: most recently submitted first
- Deduplication: by `arxiv_id`
- Class balancing: random undersample, seed = {seed}
- Collection date: {collection_date}
