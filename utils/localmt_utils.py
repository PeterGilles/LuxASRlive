# utils/localmt_utils.py — LocalMT language detection; translation via translation_service.
from __future__ import annotations

import logging

from ftlangdetect import detect as ft_detect

from utils.translation_service import translate

logger = logging.getLogger("localmt_utils")


def detect_language(text, confidence_threshold=0.5):
    """
    Detect the language of the input text using fasttext-langdetect.

    Returns:
        dict with language, confidence, reliable, and optional reason.
    """
    try:
        text = text.replace("\n", " ").strip()

        if len(text) < 10:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "reliable": False,
                "reason": "Text too short (minimum 10 characters)",
            }

        result = ft_detect(text=text, low_memory=False)

        if not result or "lang" not in result:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "reliable": False,
                "reason": "Could not detect language",
            }

        language_code = result["lang"].lower()
        confidence = result.get("score", 0.0)
        reliable = confidence >= confidence_threshold

        logger.info("Detected language: %s confidence=%.4f", language_code, confidence)

        return {
            "language": language_code,
            "confidence": round(confidence, 4),
            "reliable": reliable,
        }

    except Exception as e:
        logger.error("Language detection error: %s", e)
        return {
            "language": "unknown",
            "confidence": 0.0,
            "reliable": False,
            "reason": str(e),
        }


def localmt_translate(
    text,
    src_lang="lb",
    target_lang="en",
    context=None,
    think: bool = False,
    summary: bool = False,
    plain_language: bool = False,
):
    """
    Translate text for LocalMT (delegates to shared ``translation_service.translate``).
    """
    return translate(
        text,
        src_lang,
        target_lang,
        think=think,
        mode="localmt",
        context=context,
        summary=summary,
        plain_language=plain_language,
    )
