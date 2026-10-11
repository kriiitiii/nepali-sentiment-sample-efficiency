"""Text normalisation, PII masking, stable IDs and duplicate detection."""
from __future__ import annotations

import hashlib
import re
import unicodedata

LABELS = ("negative", "neutral", "positive")

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?977[-\s]?)?(?:9[678]\d{8}|0\d{1,2}[-\s]?\d{6,7})(?!\d)")
_MENTION = re.compile(r"(?<!\w)@[\w.]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFC normalisation and whitespace collapsing. Case is preserved."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def mask_pii(text: str) -> str:
    text = _URL.sub("<URL>", text)
    text = _EMAIL.sub("<EMAIL>", text)
    text = _PHONE.sub("<PHONE>", text)
    return _MENTION.sub("<USER>", text)


def stable_id(source: str, source_item_id: str) -> str:
    digest = hashlib.sha1(f"{source}\x1f{source_item_id}".encode("utf-8")).hexdigest()
    return f"ns-{digest[:12]}"


def dedup_key(text: str) -> str:
    """Key for exact-duplicate detection: NFC, casefolded, whitespace collapsed."""
    return normalize(text).casefold()


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = dedup_key(text)
    return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def near_duplicate_pairs(texts: list[str], threshold: float = 0.8, n: int = 5):
    """Return index pairs with character n-gram Jaccard >= threshold.

    Quadratic; intended for candidate pools of a few thousand items.
    """
    grams = [char_ngrams(t, n) for t in texts]
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if jaccard(grams[i], grams[j]) >= threshold:
                pairs.append((i, j))
    return pairs
