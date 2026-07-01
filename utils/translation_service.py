"""
Unified translation layer for LocalMT and LocalASR (Ollama /api/chat).

All user-facing translation should go through ``translate()`` so model, timeouts,
and error handling stay consistent.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import requests
from iso639 import languages

logger = logging.getLogger("translation_service")

OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

# Single model for LocalMT + LuxASR translation (override via env).
LLM_MODEL = os.environ.get(
    "LOCALMT_LLM_MODEL",
    os.environ.get("OLLAMA_LLM_MODEL", "gemma-4-9-combined-q4-k-m"),
)

_RE_STRIP_BOLD = re.compile(r"\*\*")


def resolve_language_name(code_or_name: str) -> str:
    """ISO 639-1 code or full name → display name for prompts."""
    raw = (code_or_name or "").strip()
    if not raw:
        return "Unknown"
    try:
        if len(raw) == 2:
            return languages.get(alpha2=raw.lower()).name
    except Exception:
        pass
    return raw.capitalize()


def _chat_timeout_seconds(approx_payload_chars: int = 0, *, think: bool = False) -> float:
    base = float(os.environ.get("OLLAMA_CHAT_TIMEOUT_SECONDS", "600"))
    ceiling = float(os.environ.get("OLLAMA_CHAT_TIMEOUT_MAX", "3600"))
    extra = max(0, int(approx_payload_chars)) / 100.0
    t = min(ceiling, max(30.0, base + extra))
    if think:
        mult = float(os.environ.get("OLLAMA_CHAT_THINK_TIMEOUT_MULTIPLIER", "2.5"))
        t = min(ceiling, t * mult)
    return t


def ollama_chat(
    system: str,
    user_messages: list[str],
    *,
    think: bool = False,
    temperature: Optional[float] = None,
    timeout_sec: Optional[float] = None,
    strip_bold: bool = True,
) -> str:
    """Low-level Ollama chat call. Prefer ``translate()`` for translation tasks."""
    assert isinstance(system, str)
    assert isinstance(user_messages, list)

    if temperature is None:
        temperature = float(os.environ.get("OLLAMA_CHAT_TEMPERATURE", "0.4"))

    messages = [{"role": "system", "content": system}] + [
        {"role": "assistant" if i % 2 else "user", "content": content}
        for i, content in enumerate(user_messages)
    ]

    if timeout_sec is None:
        approx = len(system) + sum(len(str(c)) for c in user_messages)
        timeout_sec = _chat_timeout_seconds(approx, think=think)

    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "think": bool(think),
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Translation service error: {str(e)}") from e

    err = data.get("error")
    if err:
        raise RuntimeError(f"Translation service error: {err}")

    if data.get("done_reason") and data["done_reason"] != "stop":
        raise RuntimeError("LLM response did not complete successfully.")

    content = (data.get("message") or {}).get("content") or ""
    content = content.strip()
    if strip_bold:
        content = _RE_STRIP_BOLD.sub("", content)
    if not content:
        raise RuntimeError("Translation service error: empty response from model.")
    return content


def _easy_read_style(target_lang: str) -> tuple[str, str]:
    code = (target_lang or "").strip().lower()[:2]
    if code == "en":
        label = "Easy Read"
    elif code == "fr":
        label = "Facile à lire et à comprendre (FALC)"
    elif code == "de":
        label = "Leichte Sprache"
    elif code == "lb":
        label = "Leichter Sprache"
    else:
        label = "Plain Language (accessible easy-read style)"
    return label, label


def _build_localmt_prompt(
    text: str,
    src_lang_name: str,
    target_lang_name: str,
    *,
    context: Optional[str],
    summary: bool,
    plain_language: bool,
) -> str:
    context_block = ""
    if context and context.strip():
        context_block = (
            "Context (for pronouns/coherence; do NOT translate):\n"
            f"{context.strip()}\n\n"
        )

    extra_tasks = []
    if summary:
        extra_tasks.append(
            f"Write a concise summary of the translation in at most 3 sentences in {target_lang_name}."
        )
    easy_read_header = ""
    if plain_language:
        easy_read_label, easy_read_header = _easy_read_style(target_lang_name)
        extra_tasks.append(
            f"Write a concise summary in {easy_read_label} in {target_lang_name}: "
            "use short sentences, common everyday words, active voice, and one main idea per sentence."
        )
    extra_block = ""
    if extra_tasks:
        extra_block = "\n".join(f"- {t}" for t in extra_tasks) + "\n\n"

    if summary or plain_language:
        section_headers = ["**Translation:**"]
        if summary:
            section_headers.append("**Summary:**")
        if plain_language:
            section_headers.append(f"**Summary ({easy_read_header}):**")
        plain_header = f"**Summary ({easy_read_header}):**" if plain_language else ""
        output_rule = (
            "Present the result in separate sections with these section headers "
            f"(use them exactly as written): {', '.join(section_headers)}. "
            f"Put the translation under **Translation:**"
            + (" and the summary under **Summary:**" if summary else "")
            + (f" and the accessible summary under {plain_header}" if plain_language else "")
            + f". All section bodies must be in {target_lang_name}."
        )
    else:
        output_rule = f"Return ONLY the translation in {target_lang_name}, with no extra commentary."

    return f"""Translate the following text from {src_lang_name} to {target_lang_name}.
{context_block}Text:
{text}

{extra_block}{output_rule}"""


def _build_asr_plain_prompt(text: str, src_lang_name: str, target_lang_name: str) -> str:
    return f"""You are a professional translator. Translate the following text from {src_lang_name} to {target_lang_name}.

IMPORTANT: Return ONLY the translated text, nothing else.

Text to translate:
{text}

Rules:
- Return ONLY the translated text in {target_lang_name}
- Preserve all timecodes in brackets [XX.XX-XX.XX]
- Preserve all speaker labels (SPEAKER_XX)
- Keep the exact same structure and formatting
- Translate only the spoken content, not the metadata
- Do not add any explanations or additional text

Translate:"""


def translate(
    text: str,
    source_lang: str,
    target_lang: str,
    *,
    think: bool = False,
    mode: str = "plain",
    context: Optional[str] = None,
    summary: bool = False,
    plain_language: bool = False,
    custom_user_prompt: Optional[str] = None,
    system: str = "You are a professional translator.",
    temperature: Optional[float] = None,
    strip_bold: bool = True,
) -> str:
    """
    Translate text using the shared Ollama backend.

    Args:
        text: Source text to translate.
        source_lang: ISO 639-1 code or language name.
        target_lang: ISO 639-1 code or language name.
        think: Enable extended model reasoning (slower).
        mode: ``plain`` | ``localmt`` | ``asr_plain`` | ``custom`` (requires custom_user_prompt).
        context: Optional surrounding context (LocalMT / subtitles).
        summary: LocalMT: add summary section.
        plain_language: LocalMT: add accessible summary section.
        custom_user_prompt: Used when mode is ``custom``.
        system: System message for Ollama.
        temperature: Override Ollama temperature.
        strip_bold: Remove ``**`` from model output.

    Returns:
        Translated text (and optional LocalMT sections).
    """
    body = (text or "").strip()
    if not body:
        return ""

    src_name = resolve_language_name(source_lang)
    tgt_name = resolve_language_name(target_lang)

    if mode == "custom":
        if not custom_user_prompt:
            raise ValueError("custom_user_prompt is required when mode='custom'")
        user_prompt = custom_user_prompt
    elif mode == "localmt":
        user_prompt = _build_localmt_prompt(
            body,
            src_name,
            tgt_name,
            context=context,
            summary=summary,
            plain_language=plain_language,
        )
        strip_bold = False  # LocalMT section headers use **
    elif mode == "asr_plain":
        user_prompt = _build_asr_plain_prompt(body, src_name, tgt_name)
    else:
        user_prompt = f"""Translate the following text from {src_name} to {tgt_name}.

Return ONLY the translation in {tgt_name}, nothing else (no headings, no quotes around the whole text).

Text:
{body}
"""

    logger.info("translate %s → %s mode=%s chars=%d", src_name, tgt_name, mode, len(body))
    return ollama_chat(
        system,
        [user_prompt],
        think=think,
        temperature=temperature,
        timeout_sec=_chat_timeout_seconds(len(user_prompt), think=think),
        strip_bold=strip_bold,
    )
