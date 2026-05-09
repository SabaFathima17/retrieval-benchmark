# Retrieval Method Benchmark: TF-IDF vs BM25 vs Hybrid

Systematic comparison of three sparse retrieval methods on SQuAD 2.0,
measuring how retrieval algorithm choice affects both passage selection
accuracy and downstream answer quality.

## Research Question

Which retrieval method produces the best passage selection on SQuAD 2.0?
Does a hybrid combination outperform individual methods?
Where do the methods differ, and why?

## Methods Compared

**TF-IDF with bigrams:** Term frequency weighted by inverse document frequency.
Uses unigrams and bigrams, stops words removed, min_df=2.
Retrieval via cosine similarity between query and passage vectors.

**BM25 Okapi:** Probabilistic extension of TF-IDF with term frequency
saturation (k1=1.5) and document length normalisation (b=0.75).
Industry standard for lexical search (used in Elasticsearch).

**Hybrid:** Linear interpolation of normalised BM25 and TF-IDF scores.
Alpha parameter controls BM25 weight. Swept from 0.0 to 1.0.

## Results (n=500 answerable questions, random seed 42)

| Method | R@1 (%) | MRR | Token F1 (%) | Exact Match (%) |
|---|---|---|---|---|
| TF-IDF | 53.4 | 0.638 | 13.0 | 0.4 |
| BM25 | 77.6 | 0.834 | 16.5 | 0.4 |
| Hybrid | 70.6 | 0.787 | 15.9 | 0.4 |

## Key Findings

**Finding 1: BM25 outperforms TF-IDF by 24 percentage points on R@1.**
Document length normalisation is the primary advantage. SQuAD 2.0 passages
range from 25 to 629 words, and TF-IDF penalises shorter passages unfairly
by not accounting for the fact that a short passage matching 3 query terms
is more informative than a long passage matching the same 3 terms.

**Finding 2: Hybrid retrieval does not beat BM25.**
The alpha sweep finds optimal alpha=1.0, meaning pure BM25 is best.
TF-IDF adds noise rather than signal when BM25 already covers lexical matching.
Hybrid methods are more useful when the methods are genuinely complementary
(e.g., dense + sparse retrieval).

**Finding 3: BM25 wins asymmetrically.**
BM25 succeeds on 134 questions where TF-IDF fails.
TF-IDF succeeds on only 13 questions where BM25 fails.
The advantage is systematic, not random variation.

**Finding 4: Both methods fail on 99 questions (19.8%).**
These are the questions where lexical overlap with the correct passage is weak.
Dense retrieval with semantic embeddings (e.g., sentence-transformers)
is the natural next step to recover these cases.

## Project Structure

```
retrieval-benchmark/
|
|-- retrieval_benchmark.ipynb    main analysis notebook
|-- README.md                    this file
|-- requirements.txt             dependencies
|
|-- outputs/
    |-- plot_01_method_comparison.png
    |-- plot_02_question_type_heatmap.png
    |-- plot_03_alpha_sweep.png
    |-- plot_04_f1_distributions.png
    |-- plot_05_method_overlap.png
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook retrieval_benchmark.ipynb
```

Place SQuAD.json in the same folder as the notebook before running.

## Tech Stack

Python, rank-bm25, scikit-learn, pandas, NumPy, Matplotlib, Seaborn
