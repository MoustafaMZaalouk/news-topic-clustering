import numpy as np
import pytest

from newsclust.data import SHA256, clean_post, ensure_data, load_all, sha256_of
from newsclust.metrics import external_scores, purity, stability
from newsclust.run import run

TOPICS = {
    0: "goal match team season player coach league score referee stadium".split(),
    1: "cpu memory disk kernel compiler network driver software server pixel".split(),
    2: "orbit rocket launch planet telescope gravity satellite lunar solar mission".split(),
}


def synthetic(n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for lab, words in TOPICS.items():
        for _ in range(n_per):
            texts.append(" ".join(rng.choice(words, size=30)))
            labels.append(lab)
    return texts, np.array(labels), ["a", "b", "c"]


def test_clean_post_strips_header_quotes_signature():
    raw = "From: x@y.z\nSubject: hi\n\nSomeone wrote:\n> old text\nnew text here\n-- \nsignature line"
    assert clean_post(raw) == "new text here"


def test_purity_and_scores_on_perfect_clustering():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert purity(y, y) == 1.0
    s = external_scores(y, np.array([5, 5, 7, 7, 9, 9]))
    assert s["ari"] == pytest.approx(1.0) and s["nmi"] == pytest.approx(1.0)
    assert purity(y, np.zeros(6, dtype=int)) == pytest.approx(2 / 6)


def test_stability_identical_runs_is_one():
    a = np.array([0, 0, 1, 1])
    assert stability([a, a, 1 - a]) == pytest.approx(1.0)


def test_synthetic_run_recovers_topics(tmp_path):
    texts, labels, names = synthetic()
    r = run(seeds=[0, 1], out_dir=tmp_path, texts=texts, labels=labels, names=names, k=3, sweep=(2, 3, 4))
    assert r["results"]["kmeans_svd100"]["ari"]["mean"] > 0.9
    assert (tmp_path / "results.json").exists() and (tmp_path / "clustering.png").exists()


def test_real_data_pinned_and_shaped():
    d = ensure_data("data")
    assert sha256_of(d / "20news-bydate.tar.gz") == SHA256
    texts, labels, names = load_all(d)
    assert len(texts) == 18846 and len(names) == 20 and len(set(labels)) == 20
    # a few posts embed a second, forwarded header in the body; the main header must be gone
    assert sum(t.startswith("From:") for t in texts) < 0.01 * len(texts)
