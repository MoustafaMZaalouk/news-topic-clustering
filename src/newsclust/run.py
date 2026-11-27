import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from .data import ensure_data, load_all
from .metrics import external_scores, stability

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["kmeans_tfidf_sparse", "kmeans_svd50", "kmeans_svd100", "kmeans_svd200", "nmf_argmax"]


def vectorize(texts):
    vec = TfidfVectorizer(stop_words="english", max_features=20000, min_df=5, max_df=0.5, sublinear_tf=True)
    return vec.fit_transform(texts), vec.get_feature_names_out()


def cluster(method, X, k, seed):
    if method == "kmeans_tfidf_sparse":
        return KMeans(n_clusters=k, n_init=1, max_iter=100, random_state=seed).fit_predict(X)
    if method.startswith("kmeans_svd"):
        dim = min(int(method.replace("kmeans_svd", "")), X.shape[1] - 1)
        Z = Normalizer(copy=False).fit_transform(TruncatedSVD(dim, random_state=seed).fit_transform(X))
        return KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(Z)
    if method == "nmf_argmax":
        W = NMF(n_components=k, init="nndsvda", max_iter=200, random_state=seed).fit_transform(X)
        return W.argmax(axis=1)
    raise ValueError(method)


def top_terms(X, terms, labels, n=8, max_clusters=20):
    out = []
    for c in np.unique(labels)[:max_clusters]:
        centroid = np.asarray(X[labels == c].mean(axis=0)).ravel()
        out.append({"cluster": int(c), "size": int((labels == c).sum()),
                    "terms": [str(terms[i]) for i in np.argsort(-centroid)[:n]]})
    return out


def run(seeds=SEEDS, subsample=None, out_dir="results", data_dir="data", texts=None, labels=None, names=None, k=20,
        sweep=(5, 10, 15, 20, 30, 40)):
    if texts is None:
        ensure_data(data_dir)
        texts, labels, names = load_all(data_dir)
    keep = [i for i, t in enumerate(texts) if len(t.split()) >= 10]
    texts = [texts[i] for i in keep]
    labels = np.asarray(labels)[keep]
    if subsample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(texts), size=subsample, replace=False)
        texts = [texts[i] for i in idx]
        labels = labels[idx]
    X, terms = vectorize(texts)
    results = {}
    seed0_labels = {}
    for m in METHODS:
        runs = []
        label_sets = []
        for s in seeds:
            lab = cluster(m, X, k, s)
            label_sets.append(lab)
            runs.append(external_scores(labels, lab))
        seed0_labels[m] = label_sets[0]
        agg = {key: {"mean": round(float(np.mean([r[key] for r in runs])), 4),
                     "std_over_seeds": round(float(np.std([r[key] for r in runs], ddof=1)) if len(runs) > 1 else 0.0, 4)}
               for key in runs[0]}
        agg["seed_stability_mean_pairwise_ari"] = round(stability(label_sets), 4)
        results[m] = agg
    # k sweep with internal silhouette (cosine on SVD-100 space, subsample for speed) vs external ARI
    Z = Normalizer(copy=False).fit_transform(TruncatedSVD(min(100, X.shape[1] - 1), random_state=0).fit_transform(X))
    rng = np.random.default_rng(0)
    sil_idx = rng.choice(len(texts), size=min(4000, len(texts)), replace=False)
    sweep_out = []
    for kk in sweep:
        if kk >= len(texts):
            continue
        lab = KMeans(n_clusters=kk, n_init=1, random_state=0).fit_predict(Z)
        sweep_out.append({"k": kk, "silhouette_cosine": round(float(silhouette_score(Z[sil_idx], lab[sil_idx])), 4),
                          "ari_vs_20_newsgroups": round(external_scores(labels, lab)["ari"], 4)})
    result = {
        "dataset": "20 Newsgroups bydate (train and test merged), headers, quoted lines and signatures removed",
        "n_documents": int(len(texts)),
        "n_label_classes": 20,
        "k": k,
        "seeds": list(seeds),
        "subsample_smoke_run": bool(subsample),
        "protocol": "unsupervised clustering with k fixed to 20; newsgroup labels are used only for scoring",
        "results": results,
        "k_sweep_svd100_kmeans_seed0": sweep_out,
        "top_terms_kmeans_svd100_seed0": top_terms(X, terms, seed0_labels["kmeans_svd100"]),
        "label_names": names,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    plot(result, out / "clustering.png")
    return result


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    names = list(result["results"])
    axes[0].barh(names, [result["results"][n]["ari"]["mean"] for n in names],
                 xerr=[result["results"][n]["ari"]["std_over_seeds"] for n in names], color="#4c78a8")
    axes[0].set_title("ARI vs newsgroup labels (k=20)", fontsize=9)
    axes[0].tick_params(labelsize=7)
    sw = result["k_sweep_svd100_kmeans_seed0"]
    ks = [s["k"] for s in sw]
    axes[1].plot(ks, [s["silhouette_cosine"] for s in sw], marker="o", label="silhouette (cosine)")
    axes[1].plot(ks, [s["ari_vs_20_newsgroups"] for s in sw], marker="s", label="ARI vs labels")
    axes[1].set_xlabel("k")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    r = run(SEEDS[: a.seeds], a.subsample, a.out)
    print("documents:", r["n_documents"])
    for m, s in r["results"].items():
        print(f"{m:22s} ARI={s['ari']['mean']:.3f} NMI={s['nmi']['mean']:.3f} purity={s['purity']['mean']:.3f} "
              f"stab={s['seed_stability_mean_pairwise_ari']:.3f}")


if __name__ == "__main__":
    main()
