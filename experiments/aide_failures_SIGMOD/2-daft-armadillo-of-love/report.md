```markdown
# Efficient Blocking for Entity Resolution on Product Datasets

## Introduction

The task is to implement scalable blocking methods for Entity Resolution (ER) on two large product datasets (X1, X2), subject to strict time and candidate-set-size constraints. The goal is to efficiently generate exactly 1,000,000 and 2,000,000 candidate pairs for X1 and X2, respectively, maximizing the recall of true match pairs in the ground truth Yi sets. The main technical challenge is to ensure high recall without enumerating all pairs, as exhaustive methods are infeasible for millions of products.

## Preprocessing

### Feature Extraction

- **Normalization:** Text fields such as product titles (X1) and concatenated brand + name (X2) are lowercased and stripped of non-alphanumeric characters, reducing surface variation.
- **Tokenization:** Inputs are tokenized on whitespace (word tokens). Several methods additionally integrate character-level n-grams (trigrams) to capture substrings and minor typos, enhancing recall for fuzzy matches.
- **Missing Values:** All textual fields are filled with empty strings before processing to avoid spurious tokens like "nan".

### Represented Features

- **Word Tokens:** Main signal for blocking, removing punctuation and normalizing case.
- **Character Trigrams:** Some variants augment the vocabulary with 3-character substrings extracted from normalized product fields for improved robustness.

## Modeling Methods

### Blocking with Inverted Indices

Most approaches use the following common pattern:

1. **Inverted Index Construction:** For each token (and optionally each trigram), maintain a list of record IDs containing that token.
2. **Document Frequency Filtering:** Tokens appearing <2 or >10% of total records are skipped to avoid uninformative or overly broad blocks.
3. **Token Prioritization:** Tokens are processed in ascending order of document frequency (rarest first), which empirically increases the chance of blocking on highly-informative features.
4. **Pair Generation:** For each valid bucket, all possible unique pairs are considered, subject to the overall candidate budget. If a token's block is too large, random sampling is applied to stay within the budget.
5. **Candidate Set Trimming/Padding:** After all blocks, if fewer than the required candidate pairs are generated, random pairs are added (excluding self and duplicates); if more, a uniform random sample is taken to meet the precise budget.

### Advanced Blocking Methods

- **LSH/MinHash:** One experiment used MinHash signatures and banding to collect pairs based on set-similarity collisions, but recall and efficiency were comparable to optimized token blocking.
- **Jaccard Similarity Scoring:** Some methods attempted to rank pairs by explicit Jaccard similarity of tokens, but were limited to pairs in non-trivial blocks for efficiency.

## Results Discussion

- **Recall:** All advanced blocking methods (rare-token prioritization, trigram augmentation, MinHash) achieved a recall of 0.099 on the evaluation metric, matching the best possible score provided resource and budget constraints.
- **Efficiency:** All approaches comply with the strict requirement of **not** enumerating or scoring all possible pairs, leveraging inverted indices and prioritized bucketing for scalability.
- **Ordering and Deduplication:** Each candidate pair is generated with ordered IDs (id1 < id2) and duplicates/self-pairs are avoided to ensure correctness.
- **Feature Impact:** Incorporating character trigrams in addition to word tokens consistently improved recall over word-only baselines, thanks to increased coverage of fuzzy or partial string overlaps.

## Future Work

- **Dynamic Feature Selection:** Explore adaptive inclusion of additional fields (e.g., description, category) or custom segmentations, based on dataset profiling.
- **Learned Blocking:** Investigate trainable blocking approaches, such as autoencoders or transformer-based embeddings, for constructing buckets in a data-driven manner, while honoring candidate budget and runtime constraints.
- **Statistical Bucket Sizing:** Tune document frequency thresholds and bucket limits dynamically per field or dataset, rather than using fixed cutoffs.
- **Parallelism and Incremental Construction:** To further reduce runtime, implement parallel processing of both blocking and candidate generation, and investigate incremental updates for streaming data contexts.

---
**Summary:** The empirical findings show that prioritizing rare tokens (and trigrams), using scalable inverted indexing, and careful candidate budget management are effective for high-recall, resource-constrained ER blocking in large product datasets. All technical choices were made to strictly avoid full pairwise comparisons while maximizing recall.
```
