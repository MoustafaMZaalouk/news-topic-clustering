# news-topic-clustering

[![ci](https://github.com/MoustafaMZaalouk/news-topic-clustering/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MoustafaMZaalouk/news-topic-clustering/actions/workflows/ci.yml)

Unsupervised topic clustering of the 20 Newsgroups posts: TF-IDF with k-means (sparse and after SVD of three sizes) and NMF, scored against the newsgroup labels, plus run-to-run stability, a sweep over k and a look at what the clusters actually contain. CPU only, about 40 seconds after a 14 MB download.

This is an AI-assisted educational project written by a student. It is not a topic-modelling product and not client work. No neural embeddings are used (the build machine had no GPU or torch), so this says nothing about what embeddings would do.

## Setup
- 18,605 posts (train and test of the `bydate` archive merged), after removing headers, quoted lines and signatures and dropping posts under 10 words.
- TF-IDF, 20,000 terms, English stop words. k is fixed to 20 to match the number of newsgroups. The newsgroup labels are used only for scoring, never for fitting.
- 5 seeds per method. Scores are mean (std over seeds).

## Results

| method | ARI | NMI | purity | seed stability (mean pairwise ARI) |
|---|---|---|---|---|
| `kmeans_tfidf_sparse` | 0.130 (0.009) | 0.331 (0.016) | 0.335 (0.018) | 0.566 |
| `kmeans_svd50` | 0.203 (0.012) | 0.361 (0.009) | 0.382 (0.011) | 0.655 |
| `kmeans_svd100` | 0.205 (0.006) | 0.377 (0.003) | 0.397 (0.014) | 0.638 |
| `kmeans_svd200` | 0.174 (0.009) | 0.348 (0.022) | 0.354 (0.023) | 0.574 |
| `nmf_argmax` | 0.159 (0.009) | 0.308 (0.014) | 0.319 (0.013) | 0.743 |

Sweep over k (k-means on SVD-100, seed 0; silhouette on a 4,000-post subsample, cosine):

| k | silhouette | ARI vs 20 labels |
|---|---|---|
| 5 | 0.039 | 0.079 |
| 10 | 0.049 | 0.134 |
| 15 | 0.054 | 0.144 |
| 20 | 0.064 | 0.197 |
| 30 | 0.074 | 0.210 |
| 40 | 0.086 | 0.217 |

Plots: `results/clustering.png`. Raw numbers and all 20 clusters' top terms: `results/results.json`.

## How to read this
- Agreement with the labels is modest. The best setting (SVD-100 then k-means) has ARI about 0.20 and NMI about 0.38. Many newsgroups overlap in topic (several `comp.*`, `talk.politics.*`, `religion`), so a ceiling of 1.0 is not a fair target.
- SVD before k-means helped over sparse k-means (ARI 0.20 against 0.13). SVD with 200 components was worse than with 50 or 100. This is one corpus and fixed other settings, so it is a description, not a rule.
- The clusterings differ between seeds (stability 0.57 to 0.74). NMF is the most stable but scores lower against the labels.
- The silhouette score rises with k and does not point to 20, so it is not usable for choosing k here.
- Some clusters group by posting style or email domain, not topic. For example cluster 3 is dominated by `edu article cs uiuc cso news cmu` and cluster 9 by `com article netcom hp att`. Headers and quotes were removed, but sender signatures and domains remain in the body in many posts. Other clusters are clearly topical:

| cluster | posts | top terms |
|---|---|---|
| 0 | 410 | turkish armenian armenians armenia turks 12 |
| 2 | 408 | israel israeli jews arab arabs jewish |
| 3 | 2,118 | edu article cs uiuc cso news |
| 6 | 1,077 | game team games year hockey season |
| 7 | 496 | drive scsi drives disk hard ide |
| 9 | 1,611 | com article netcom hp don att |
| 13 | 1,104 | god jesus bible christ christian christians |
| 15 | 350 | cc edu au article monash columbia |
| 17 | 533 | key clipper encryption chip keys government |
| 19 | 599 | nasa gov space orbit shuttle jpl |

- Cluster summaries come from the mean TF-IDF vector of each cluster. They are a quick look, not a coherence measurement.

## Data
Not included. `make run` downloads `20news-bydate.tar.gz` from the original distribution page (qwone.com/~jason/20Newsgroups) over HTTP and checks its SHA-256 (`8f1b25...610`) before extracting. Dataset collected by Ken Lang and distributed by Jason Rennie. Check that page for terms of use. Code here is MIT only.

## Run
```
pip install -r requirements.txt
make test     # 5 tests; downloads the dataset once
make smoke    # 3,000-post, 1-seed run
make run      # full run, rewrites results/
```

## Layout
`src/newsclust/` data, metrics, run | `tests/` | `results/` committed output of the last full run | `.github/workflows/ci.yml`

## License
MIT, code only. Copyright Moustafa Mouhamed Zaalouk.
