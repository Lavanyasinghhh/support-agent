"""TF-IDF retriever over the help-centre articles.

Each article is split into paragraphs ("chunks"). A query is matched against
all chunks with cosine similarity, so we can return the most relevant passages
and a score that tells the agent how confident the match is.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN = re.compile(r"[a-z0-9$]+")
STOP = {"a", "an", "the", "is", "are", "do", "does", "you", "your", "i", "my", "me", "to",
        "of", "for", "in", "on", "it", "can", "and", "or", "if", "how", "what", "we", "be"}


def tokenize(text: str) -> list[str]:
    """Lower-case, drop stop words, and strip simple suffixes so that
    "refunds", "refunded" and "refund" all match."""
    out = []
    for tok in TOKEN.findall(text.lower()):
        if tok in STOP:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(tok) > 4 and tok.endswith(suffix):
                tok = tok[: -len(suffix)]
                break
        out.append(tok)
    return out


@dataclass
class Chunk:
    source: str   # article file name, e.g. "returns.md"
    title: str    # article heading
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


def load_chunks(kb_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        title = lines[0].lstrip("# ").strip()
        body = "\n".join(lines[1:])
        for para in body.split("\n\n"):
            para = para.strip()
            if para:
                chunks.append(Chunk(source=path.name, title=title, text=para))
    return chunks


class Retriever:
    def __init__(self, kb_dir: Path):
        self.chunks = load_chunks(kb_dir)
        # Include the title in each chunk so "warranty" matches warranty paragraphs.
        docs = [f"{c.title}. {c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in best]
