"""Japanese tokenizer for vecmemori FTS5 integration.

Wraps fugashi (MeCab) for Japanese morphological analysis.
Supports explicit language control: "auto", "ja", "en".

Usage:
    >>> tokenize("ユーザーはダークモードを好む")
    'ユーザー は ダーク モード を 好む'
    >>> tokenize("User prefers dark mode", language="en")
    'User prefers dark mode'
    >>> tokenize("tokenmaxxing は重要", language="auto")
    'tokenmaxxing は 重要'
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

# FTS5 query operators — skip tokenization when present
_FTS5_OP_PATTERN = re.compile(
    r'[\"()*^]|'
    r'\b(?:AND|OR|NOT)\b'
)

# Language modes
LANG_AUTO = "auto"
LANG_JA = "ja"
LANG_EN = "en"

# Japanese character ranges for auto-detection
_JA_RANGES = [
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility
    (0xFF66, 0xFF9F),  # Half-width Katakana
]


def _detect_language(text: str) -> str:
    """Auto-detect: >=3% Japanese characters → 'ja', else 'en'."""
    if not text:
        return LANG_EN
    ja_chars = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cp = ord(ch)
        for lo, hi in _JA_RANGES:
            if lo <= cp <= hi:
                ja_chars += 1
                break
    if total == 0:
        return LANG_EN
    return LANG_JA if (ja_chars / total) >= 0.03 else LANG_EN


def _get_tagger():
    """Lazy-load the fugashi Tagger singleton."""
    global _tagger
    if _tagger is None and _HAS_FUGASHI:
        try:
            _tagger = Tagger("-Owakati")
            _tagger.parse("テスト")  # warm-up
            logger.debug("fugashi Tagger initialized successfully")
        except Exception as exc:
            logger.warning("fugashi Tagger initialization failed: %s", exc)
            return None
    return _tagger


def has_tokenizer() -> bool:
    """Check if fugashi is available and initialized."""
    return _get_tagger() is not None


def tokenize(text: str, language: str = LANG_AUTO) -> str:
    """Tokenize text for FTS5 indexing.

    Args:
        text: Input text.
        language: "auto" (detect), "ja" (force fugashi), "en" (native unicode61).

    Returns:
        Space-separated tokens for FTS5, or original text if fugashi unavailable.
    """
    if not text:
        return text

    lang = (language or LANG_AUTO).lower()
    if lang == LANG_AUTO:
        lang = _detect_language(text)
    if lang == LANG_EN:
        # unicode61 handles English word boundaries natively
        return text

    # Japanese mode
    if lang == LANG_JA:
        tagger = _get_tagger()
        if tagger is None:
            return text
        try:
            result = tagger.parse(text).strip()
            return result if result else text
        except Exception as exc:
            logger.debug("fugashi tokenization failed: %s", exc)
            return text

    return text


def tokenize_query(query: str, language: str = LANG_AUTO) -> str:
    """Tokenize a search query for FTS5 MATCH.

    Preserves FTS5 operators (AND, OR, NOT, quotes, etc.) when detected.

    Args:
        query: Raw search query.
        language: "auto", "ja", or "en".

    Returns:
        Tokenized query or original if operators detected.
    """
    if not query:
        return query
    if _FTS5_OP_PATTERN.search(query):
        return query
    return tokenize(query, language=language)
