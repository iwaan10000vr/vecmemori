"""Japanese tokenizer for vecmemori FTS5 integration.

Wraps fugashi (MeCab) for Japanese morphological analysis.
Gracefully falls back to no-op when fugashi is not installed.

Usage:
    >>> from vecmemori._tokenizer import tokenize, has_tokenizer
    >>> tokenize("ユーザーはダークモードを好む")
    'ユーザー は ダーク モード を 好む'
    >>> has_tokenizer()
    True
"""

import logging
import re

logger = logging.getLogger(__name__)

_HAS_FUGASHI = False
_tagger = None

try:
    from fugashi import Tagger

    _HAS_FUGASHI = True
except ImportError:
    Tagger = None  # type: ignore[assignment]

# FTS5 query operators that should not be tokenized
# If a query contains any of these, it's treated as an advanced FTS5 query
_FTS5_OP_PATTERN = re.compile(
    r'["()*^]|'  # quotes, parens, wildcards, caret
    r'\b(?:AND|OR|NOT)\b'  # boolean operators
)


def _get_tagger():
    """Lazy-load the fugashi Tagger singleton."""
    global _tagger
    if _tagger is None and _HAS_FUGASHI:
        try:
            _tagger = Tagger("-Owakati")
            # Warm up with a short parse to verify it works
            _tagger.parse("テスト")
            logger.debug("fugashi Tagger initialized successfully")
        except Exception as exc:
            logger.warning("fugashi Tagger initialization failed: %s", exc)
            return None
    return _tagger


def has_tokenizer() -> bool:
    """Check if fugashi is available and initialized."""
    return _get_tagger() is not None


def tokenize(text: str) -> str:
    """Tokenize Japanese text into space-separated tokens for FTS5.

    Falls back to original text if fugashi is not available.
    Non-Japanese text (English, digits, etc.) is passed through unchanged.

    Args:
        text: Input text to tokenize.

    Returns:
        Space-separated tokenized text, or original text if fugashi unavailable.
    """
    if not text:
        return text
    tagger = _get_tagger()
    if tagger is None:
        return text
    try:
        result = tagger.parse(text).strip()
        return result if result else text
    except Exception as exc:
        logger.debug("fugashi tokenization failed: %s", exc)
        return text


def tokenize_query(query: str) -> str:
    """Tokenize a search query for FTS5 MATCH.

    Detects FTS5 operators — if present, the query is passed through
    without tokenization to preserve operator syntax.

    Args:
        query: Raw search query from user or agent.

    Returns:
        Tokenized query for FTS5 MATCH, or original if operators are detected.
    """
    if not query:
        return query
    if _FTS5_OP_PATTERN.search(query):
        # Query contains FTS5 operators — pass through unchanged
        return query
    return tokenize(query)
