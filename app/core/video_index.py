"""
Video search index built from WLASL_v0.3.json.

Search tiers (applied in order, results merged by best score):
  1. Exact match          — gloss == normalized query
  2. Morphological        — lemmatized query variant matches a gloss
  3. Prefix               — gloss starts with query
  4. Fuzzy                — edit-distance similarity (rapidfuzz if available, else difflib)
  5. Synonym expansion    — WordNet synonyms of each query word, then tiers 1-4 on each

rapidfuzz and nltk are optional; the index degrades gracefully without them.
"""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("cslr.video_index")

# ---------------------------------------------------------------------------
# Optional: rapidfuzz  (much better fuzzy than difflib)
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_proc
    _HAS_RAPIDFUZZ = True
    log.info("rapidfuzz available — using WRatio fuzzy search")
except ImportError:
    _HAS_RAPIDFUZZ = False
    log.info("rapidfuzz not installed — falling back to difflib")

# ---------------------------------------------------------------------------
# Optional: NLTK WordNet  (semantic synonym expansion)
# ---------------------------------------------------------------------------
_wordnet = None
_HAS_WORDNET = False
try:
    import nltk
    from nltk.corpus import wordnet as _wn

    def _ensure_wordnet() -> bool:
        try:
            _wn.synsets("book")
            return True
        except LookupError:
            pass
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            _wn.synsets("book")
            return True
        except Exception:
            return False

    if _ensure_wordnet():
        _wordnet = _wn
        _HAS_WORDNET = True
        log.info("NLTK WordNet available — semantic synonym expansion enabled")
    else:
        log.info("NLTK WordNet data unavailable — synonym expansion disabled")
except ImportError:
    log.info("NLTK not installed — synonym expansion disabled")

# ---------------------------------------------------------------------------
# Simple morphological normaliser (no external deps)
# Handles common English inflections so "books" → "book", "drinking" → "drink"
# ---------------------------------------------------------------------------
_SUFFIX_RULES: List[tuple[str, str]] = [
    ("ying", "ie"), ("ving", "ve"),
    ("ies",  "y"),  ("ves",  "f"),
    ("oes",  "o"),  ("ses",  "s"),
    ("xes",  "x"),  ("zes",  "z"),
    ("ing",  ""),   ("ied",  "y"),
    ("es",   ""),   ("ed",   ""),
    ("s",    ""),
]

def _lemmatize(word: str) -> List[str]:
    """Return candidate base forms for a single word (including the word itself)."""
    candidates = {word}
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            base = word[: -len(suffix)] + replacement
            if len(base) >= 2:
                candidates.add(base)
    return list(candidates)


def _wordnet_synonyms(word: str) -> List[str]:
    """
    Return synonyms restricted to the same dominant POS as the query word.
    Filters out function words / very short lemmas (len < 3) to avoid
    spurious matches like 'in', 'to', 'do' appearing in synonym paths.
    """
    if not _HAS_WORDNET or _wordnet is None:
        return []

    synsets = _wordnet.synsets(word)
    if not synsets:
        return []

    dominant_pos = synsets[0].pos()  # most common POS for this word

    out: set[str] = set()
    for syn in synsets:
        if syn.pos() != dominant_pos:
            continue
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word and len(name) >= 3:
                out.add(name)
    return list(out)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class VideoIndex:
    """
    Immutable index built once at startup.
    Thread-safe for concurrent read-only search calls.
    """

    def __init__(
        self,
        json_path: str = "archive/WLASL_v0.3.json",
        video_dir: str = "archive/videos",
        url_prefix: str = "/archive-videos",
    ) -> None:
        self._url_prefix = url_prefix
        self._gloss_to_ids: Dict[str, List[str]] = {}
        self._glosses: List[str] = []
        self._build(json_path, video_dir)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, json_path: str, video_dir: str) -> None:
        vid_dir = Path(video_dir)
        if not vid_dir.is_dir():
            log.warning("video_dir %s does not exist — video index empty", video_dir)
            return

        local_ids = {p.stem for p in vid_dir.glob("*.mp4")}
        log.info("Found %d local .mp4 files", len(local_ids))

        json_file = Path(json_path)
        if not json_file.is_file():
            log.warning("WLASL JSON not found at %s", json_path)
            return

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            gloss = entry["gloss"].strip().lower()
            ids = [
                inst["video_id"]
                for inst in entry["instances"]
                if inst["video_id"] in local_ids
            ]
            if ids:
                self._gloss_to_ids[gloss] = ids

        self._glosses = sorted(self._gloss_to_ids)
        log.info(
            "VideoIndex built: %d glosses, %d total instances",
            len(self._glosses),
            sum(len(v) for v in self._gloss_to_ids.values()),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _video_urls(self, gloss: str, max_videos: int = 8) -> List[str]:
        ids = self._gloss_to_ids.get(gloss, [])[:max_videos]
        return [f"{self._url_prefix}/{vid}.mp4" for vid in ids]

    def _fuzzy_candidates(self, term: str, cutoff_pct: int = 60, top: int = 15) -> List[tuple[str, float]]:
        """
        Return (gloss, 0-1 score) pairs from fuzzy matching.

        Length-ratio guard: a candidate gloss must have length in [0.5×, 2.5×]
        of the query length.  This stops WRatio's partial-ratio sub-scorer
        from inflating short glosses (e.g. the single-letter ASL alphabet
        signs "d", "i", "k") when they appear as substrings of the query.
        """
        q_len = len(term)
        min_len = max(3, int(q_len * 0.55))
        max_len = max(q_len + 4, int(q_len * 2.5))

        if _HAS_RAPIDFUZZ:
            hits = _rf_proc.extract(
                term, self._glosses,
                scorer=_rf_fuzz.WRatio,
                limit=top * 4,          # over-fetch before length filter
                score_cutoff=cutoff_pct,
            )
            return [
                (g, score / 100.0)
                for g, score, _ in hits
                if min_len <= len(g) <= max_len
            ][:top]
        else:
            hits = difflib.get_close_matches(term, self._glosses, n=top * 4, cutoff=cutoff_pct / 100.0)
            scored = []
            for g in hits:
                if not (min_len <= len(g) <= max_len):
                    continue
                ratio = difflib.SequenceMatcher(None, term, g).ratio()
                scored.append((g, ratio))
            return sorted(scored, key=lambda x: -x[1])[:top]

    def _add_result(
        self,
        seen: Dict[str, Dict],
        gloss: str,
        match_type: str,
        score: float,
    ) -> None:
        if gloss not in self._gloss_to_ids:
            return
        if gloss not in seen or seen[gloss]["score"] < score:
            seen[gloss] = {
                "gloss": gloss,
                "match_type": match_type,
                "score": round(score, 4),
                "videos": self._video_urls(gloss),
                "video_count": len(self._gloss_to_ids.get(gloss, [])),
            }

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 12) -> List[Dict]:
        """
        Multi-tier search.  Returns up to `limit` results sorted by score desc.
        Each result: {gloss, match_type, score, videos, video_count}
        """
        q = query.strip().lower()
        if not q:
            return []

        seen: Dict[str, Dict] = {}
        add = self._add_result  # shortcut

        # ── Tier 1: exact ──────────────────────────────────────────────
        if q in self._gloss_to_ids:
            add(seen, q, "exact", 1.0)

        # ── Tier 2: morphological variants ─────────────────────────────
        for base in _lemmatize(q):
            if base != q:
                if base in self._gloss_to_ids:
                    add(seen, base, "morphological", 0.93)
                # also prefix-match the base
                for g in self._glosses:
                    if g.startswith(base) and g != base:
                        add(seen, g, "prefix", 0.78)

        # ── Tier 3: prefix (on original query) ─────────────────────────
        for g in self._glosses:
            if g.startswith(q) and g != q:
                add(seen, g, "prefix", 0.82)

        # ── Tier 4: fuzzy (on original query) ──────────────────────────
        for g, ratio in self._fuzzy_candidates(q, cutoff_pct=62, top=20):
            if g not in seen:
                add(seen, g, "fuzzy", round(ratio * 0.74, 4))

        # ── Tier 5: semantic synonym expansion via WordNet ──────────────
        expanded_syns: set[str] = set()
        for word in q.split():
            for syn in _wordnet_synonyms(word):
                expanded_syns.add(syn)
                for base in _lemmatize(syn):
                    if len(base) >= 3:          # guard: drop degenerate lemmas like "in"
                        expanded_syns.add(base)

        for syn in expanded_syns:
            if len(syn) < 3:
                continue
            # exact match on synonym
            if syn in self._gloss_to_ids:
                add(seen, syn, "synonym", 0.76)
            # fuzzy on synonym (high cutoff — synonyms have different spelling)
            for g, ratio in self._fuzzy_candidates(syn, cutoff_pct=78, top=5):
                if g not in seen:
                    add(seen, g, "synonym", round(ratio * 0.70, 4))

        results = sorted(seen.values(), key=lambda r: (-r["score"], -r["video_count"]))
        return results[:limit]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def gloss_count(self) -> int:
        return len(self._gloss_to_ids)

    @property
    def total_video_count(self) -> int:
        return sum(len(v) for v in self._gloss_to_ids.values())

    @property
    def nlp_info(self) -> Dict:
        return {
            "rapidfuzz": _HAS_RAPIDFUZZ,
            "wordnet": _HAS_WORDNET,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[VideoIndex] = None


def init_video_index(
    json_path: str = "archive/WLASL_v0.3.json",
    video_dir: str = "archive/videos",
) -> VideoIndex:
    global _instance
    _instance = VideoIndex(json_path, video_dir)
    return _instance


def get_video_index() -> Optional[VideoIndex]:
    return _instance
