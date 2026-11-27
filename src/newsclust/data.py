"""Download, verify and load the 20 Newsgroups 'bydate' archive (Ken Lang / Jason Rennie's distribution)."""
import hashlib
import re
import tarfile
import urllib.request
from pathlib import Path

URL = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
SHA256 = "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610"
DATA_DIR = Path("data")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    if (data_dir / "20news-bydate-train").is_dir() and (data_dir / "20news-bydate-test").is_dir():
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tgz = data_dir / "20news-bydate.tar.gz"
    urllib.request.urlretrieve(URL, tgz)
    got = sha256_of(tgz)
    if got != SHA256:
        tgz.unlink()
        raise RuntimeError(f"checksum mismatch: expected {SHA256}, got {got}")
    with tarfile.open(tgz) as tf:
        members = tf.getmembers()
        for m in members:
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
                raise RuntimeError(f"unsafe archive member: {m.name}")
        tf.extractall(data_dir, members=members)
    return data_dir


def clean_post(text):
    """Drop the header block, quoted lines and signature so topics come from the body text only."""
    parts = re.split(r"\n[ \t]*\n", text, maxsplit=1)
    body = parts[1] if len(parts) == 2 else text
    body = re.split(r"\n-- ?\n", body)[0]
    lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith(">") and not ln.rstrip().endswith("wrote:")]
    return "\n".join(lines).strip()


def load_all(data_dir=DATA_DIR):
    """Return (texts, labels, label_names) for train and test merged, header-stripped."""
    data_dir = Path(data_dir)
    names = sorted(p.name for p in (data_dir / "20news-bydate-train").iterdir() if p.is_dir())
    texts, labels = [], []
    for split in ("20news-bydate-train", "20news-bydate-test"):
        for li, name in enumerate(names):
            for f in sorted((data_dir / split / name).iterdir()):
                texts.append(clean_post(f.read_text(encoding="latin-1")))
                labels.append(li)
    return texts, labels, names
