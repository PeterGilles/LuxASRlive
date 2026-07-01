###################################################################################################################################
###################################################################################################################################
####                                                                                                                           ####
####   Title   : LuxASR WebSocket Gateway (FastAPI) - Version 2.3.0                                                            ####
####   Version : 2.3.0                                                                                                         ####
####   Author  : Léopold Hillah                                                                                                ####
####   Date    : 16.05.2026                                                                                                    ####
####   Copyrights: Sproochtek S.à.r.l.-s                                                                                       ####
####                                                                                                                           ####
####   Lightweight WebSocket gateway for real-time ASR using FastAPI + Uvicorn.                                               ####
####   Version 2.0 adds context management for improved transcription continuity.                                              ####
####   Version 2.1 adds real-time translation support.                                                                         ####
####                                                                                                                           ####
####   New in v2.0:                                                                                                            ####
####   - Context/prompt passing to Whisper for continuity                                                                      ####
####   - Intelligent context reset based on silence and punctuation                                                            ####
####   - Repetition detection to prevent hallucination loops                                                                   ####
####   - Configurable context management settings                                                                              ####
####                                                                                                                           ####
####   New in v2.0.1:                                                                                                          ####
####   - ASR optimization parameters (vad_filter=false, beam_size=3)                                                           ####
####   - Faster inference by avoiding double VAD processing                                                                    ####
####                                                                                                                           ####
####   New in v2.0.2:                                                                                                          ####
####   - Speech validation before API calls (skip empty/silent chunks)                                                         ####
####   - Prevents 500 errors from sending silence to Whisper                                                                   ####
####   - Configurable energy and speech ratio thresholds                                                                       ####
####                                                                                                                           ####
####   New in v2.0.3:                                                                                                          ####
####   - Silent skip: no message sent to client for empty chunks                                                               ####
####   - Processing indicator: sends "processing" message before transcription                                                 ####
####   - Periodic sending: sends chunks every N seconds during continuous speech                                               ####
####   - Reduced silence threshold (0.8s) for faster response                                                                  ####
####   - Fixed recording restart: session stays open after stop, can restart without reconnect                                 ####
####                                                                                                                           ####
####   New in v2.0.4:                                                                                                          ####
####   - Overlapping audio chunks to avoid cutting words at boundaries                                                         ####
####   - Text reconciliation to remove duplicate text from overlapping regions                                                 ####
####   - Configurable overlap duration and reconciliation settings                                                             ####
####                                                                                                                           ####
####   New in v2.0.5:                                                                                                          ####
####   - VAD-aware periodic sending: waits for natural pauses before cutting                                                   ####
####   - Configurable minimum silence for periodic send (periodic_min_silence)                                                 ####
####   - Timeout fallback if no pause detected (periodic_max_wait)                                                             ####
####   - New send reasons: periodic_pause (natural), periodic_timeout (forced)                                                 ####
####                                                                                                                           ####
####   New in v2.1.0:                                                                                                          ####
####   - Real-time translation support via local localmt_translate function (Ollama)                                             ####
####   - Configurable translation target language                                                                              ####
####   - Translation displayed alongside transcription                                                                         ####
####   - Accumulated translation tracking                                                                                      ####
####                                                                                                                           ####
####   New in v2.2.0:                                                                                                          ####
####   - Timestamp-based overlap reconciliation (word_timestamps=true API param)                                               ####
####   - Fuzzy text-based reconciliation fallback (SequenceMatcher, prefix match)                                              ####
####   - Last-word withholding: final word of each chunk held back one chunk                                                   ####
####     and emitted only after the next chunk confirms the boundary is stable.                                                ####
####     Eliminates partial/cut words at chunk boundaries (e.g. "et"→"e" class).                                              ####
####   - Inspired by Local Agreement (whisper_streaming) and CIF (SimulStreaming)                                              ####
####                                                                                                                           ####
####   New in v2.2.1:                                                                                                          ####
####   - PostProcessor: language-aware text cleanup applied to each emitted segment ####
####   - Elision fix: "d' Word" → "d'Word" (Luxembourgish/French articles)          ####
####   - Spurious full-stop removal: chunk-boundary "oder." before lowercase        ####
####     continuation → "oder"; leading-dot artefacts ".gesot" → "gesot"            ####
####                                                                                                                           ####
####   New in v2.3.0:                                                                                                          ####
####   - Timestamp tolerance: overlap boundary extended by overlap_timestamp_tolerance  ####
####     (default 0.15s) to absorb Whisper's ±150ms word-timing variance.           ####
####   - Hybrid reconciliation fallback: if timestamp path removes 0 words, the     ####
####     text-based path is tried as a safety net.                                   ####
####   - First-word guard: _apply_last_word_withholding checks if the first word of ####
####     the new chunk fuzzy-matches the withheld word; if so, drops the duplicate. ####
####   - RepetitionClassifier: 3-signal voting (position, temporal gap, context)    ####
####     distinguishes boundary artefacts from genuine speech repetitions.           ####
####   - Increased default chunk_overlap_duration: 0.5s → 0.8s for wider margin.   ####
####   - ASR response parser: handles diarisation list format from production API.  ####
####                                                                                                                           ####
###################################################################################################################################
###################################################################################################################################

import io
import re
import wave
import json
import time
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
from collections import deque
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

# Import local translation utility (v2.1.0)
from utils.localmt_utils import localmt_translate
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# =============================================================================
# Configuration
# =============================================================================

base_path = "<application user home dir>"
log_path = f"{base_path}/logs"

@dataclass
class GatewayConfig:
    """Configuration for the WebSocket Gateway"""
    
    # ASR API endpoint (via Nginx on localhost)
    asr_api_url: str = "<target ASR endpoint>" # example "http://127.0.0.1/v2/live"
    asr_timeout: float = 120.0
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit audio
    
    # Buffering settings
    min_chunk_duration: float = 1.0
    max_chunk_duration: float = 30.0
    silence_threshold: float = 0.8      # Reduced from 1.5s for faster response
    
    # ==========================================================================
    # NEW in v2.0.5: VAD-Aware Periodic Sending
    # ==========================================================================
    
    # Send chunks every N seconds during continuous speech (0 = disabled)
    periodic_send_interval: float = 5.0
    
    # Minimum silence (in seconds) required before periodic send
    # This prevents cutting audio mid-word
    periodic_min_silence: float = 0.3
    
    # Maximum extra wait time (in seconds) for a natural pause after interval
    # If no pause occurs within this time, send anyway (with overlap to help)
    periodic_max_wait: float = 2.0
    
    # ==========================================================================
    # NEW in v2.0.4: Chunk Overlap Settings
    # ==========================================================================
    
    # Enable overlapping chunks to avoid cutting words at boundaries
    enable_chunk_overlap: bool = True
    
    # Duration of overlap in seconds (audio from end of previous chunk 
    # prepended to the next chunk).
    # Increased from 0.5 → 0.8 in v2.3.0: wider margin means more boundary
    # words fall comfortably inside the overlap window.
    chunk_overlap_duration: float = 0.8

    # Timestamp tolerance added to the overlap boundary cutoff (v2.3.0).
    # Whisper's word-level timestamps can be off by ±100-200ms; extending
    # the discard window by this amount catches words that straddle the edge.
    overlap_timestamp_tolerance: float = 0.15

    # Use word-level timestamps from ASR for overlap reconciliation when
    # available. More accurate than text-based matching — falls back to
    # text-based reconciliation if the API does not return timestamps.
    use_word_timestamps: bool = True

    # Repetition classifier (NEW in v2.3.0)
    # -------------------------------------------------------------------------
    # After reconciliation, a 3-signal voting classifier checks whether the
    # first word of the new chunk is a boundary artefact (duplicate) or a
    # genuine speech repetition.  Requires at least 2-of-3 signals to agree
    # before removing the word, so real repetitions ("nee nee", counting
    # sequences like "zwou Wochen, dräi Wochen") are preserved.
    enable_repetition_classifier: bool = True
    repetition_genuine_gap_threshold: float = 0.20   # seconds

    # Last-word withholding (NEW in v2.2.0)
    # -------------------------------------------------------------------------
    # The final decoded word of any audio chunk is statistically unreliable:
    # Whisper was forced to decode it from potentially incomplete audio right
    # at the chunk boundary.  Inspired by the CIF mechanism in SimulStreaming
    # and the Local Agreement principle in whisper_streaming, we withhold that
    # last word from the client until the next chunk's transcription confirms
    # the boundary is stable.
    #
    # Behaviour:
    #   chunk N  → text "well et"   → emit "well",    hold "et"
    #   chunk N+1→ text "ganz datt" → emit "et ganz", hold "datt"
    #   finalize → remaining text   → emit "datt ..."  (full flush)
    #
    # Only active when the post-reconciliation text has ≥ 2 words.
    # Single-word chunks and the final (finalize) chunk are always emitted in full.
    enable_last_word_withholding: bool = True
    
    # Number of words to check for overlap reconciliation (increased from 5 to 8)
    overlap_reconcile_words: int = 8
    
    # VAD settings
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 100
    
    # Default transcription settings
    default_language: str = "lb"
    diarization: str = "Disabled"
    output_format: str = "text"
    
    # ==========================================================================
    # NEW in v2.0: Real-time ASR Optimization Settings
    # ==========================================================================
    
    # Disable VAD filter in Whisper (gateway already does VAD)
    asr_vad_filter: bool = False
    
    # Lower beam size for faster inference (1-5, lower = faster)
    asr_beam_size: int = 3
    
    # ==========================================================================
    # NEW in v2.0.3: Speech Validation Settings (skip empty chunks)
    # ==========================================================================
    
    # Enable skipping chunks that contain no speech
    skip_empty_chunks: bool = True
    
    # Minimum ratio of audio that should contain speech (0.0-1.0)
    # E.g., 0.05 means at least 5% of the chunk should have speech
    min_speech_ratio: float = 0.05
    
    # Minimum RMS energy threshold (0-32768 for 16-bit audio)
    # Audio below this is considered silence
    min_audio_energy: float = 100.0
    
    # Supported languages
    supported_languages: List[str] = field(default_factory=lambda: [
        "lb", "en", "fr", "de", "es", "pt"
    ])
    
    # VAD model path
    vad_model_path: str = f"{base_path}/.../hub/snakers4_silero-vad_master"
    
    # Paths
    templates_dir: str = "templates"
    static_dir: str = "static"
    
    # ==========================================================================
    # NEW in v2.0: Context Management Settings
    # ==========================================================================
    
    # Enable/disable context passing to Whisper
    use_context: bool = True
    
    # Maximum tokens to send as context (roughly 1 token ≈ 0.75 words)
    max_context_tokens: int = 80
    
    # Reset context after this much silence (seconds)
    context_reset_silence: float = 4.0
    
    # Reset context after sentence-final punctuation + this much silence
    context_reset_punctuation_silence: float = 2.0
    
    # Sentence-ending punctuation marks
    sentence_end_punctuation: str = ".!?"
    
    # Enable repetition detection to prevent hallucination loops
    detect_repetition: bool = True
    
    # Number of recent segments to check for repetition
    repetition_window: int = 3
    
    # Similarity threshold for repetition detection (0.0-1.0)
    repetition_threshold: float = 0.8
    
    # ==========================================================================
    # NEW in v2.1.0: Translation Settings
    # ==========================================================================
    
    # Translation uses local localmt_translate function (connects to Ollama)
    # Fallback API URL (not used by default, kept for reference)
    translation_api_url: str = "<translation API endpoint>" # Claude, Gemini, ChatGPT, Llama, Qwen ...
    
    # Supported target languages for translation
    translation_target_languages: List[str] = field(default_factory=lambda: [
        "en", "fr", "de", "lb", "pt", "es"
    ])


# Global config
config = GatewayConfig()

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{log_path}/ws_gateway.log")
    ]
)
logger = logging.getLogger("ws_gateway")

# =============================================================================
# Silero VAD (Singleton, CPU-only)
# =============================================================================

class SileroVAD:
    """Silero Voice Activity Detection - CPU only, lightweight"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.get_speech_timestamps = None
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """Load Silero VAD model (CPU only)"""
        try:
            local_path = Path(config.vad_model_path)
            
            if local_path.exists():
                logger.info(f"Loading Silero VAD from: {local_path}")
                self.model, utils = torch.hub.load(
                    str(local_path),
                    model='silero_vad',
                    source='local',
                    onnx=False
                )
            else:
                logger.info("Loading Silero VAD from torch hub...")
                self.model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
            
            self.model = self.model.cpu()
            self.get_speech_timestamps = utils[0]
            logger.info("Silero VAD loaded successfully (CPU)")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            self.model = None
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Dict]:
        """Detect speech segments in audio"""
        if self.model is None:
            return [{"start": 0, "end": len(audio)}]
        
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0
            
            audio_tensor = torch.from_numpy(audio)
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=config.vad_threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=config.vad_min_speech_duration_ms,
                min_silence_duration_ms=config.vad_min_silence_duration_ms
            )
            
            return speech_timestamps
            
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return [{"start": 0, "end": len(audio)}]
    
    def has_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio contains speech"""
        timestamps = self.detect_speech(audio, sample_rate)
        return len(timestamps) > 0
    
    def get_speech_ratio(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Get the ratio of audio that contains speech (0.0 to 1.0).
        
        Returns:
            Float between 0.0 (no speech) and 1.0 (all speech)
        """
        timestamps = self.detect_speech(audio, sample_rate)
        
        if not timestamps:
            return 0.0
        
        total_samples = len(audio)
        if total_samples == 0:
            return 0.0
        
        speech_samples = sum(ts["end"] - ts["start"] for ts in timestamps)
        return min(1.0, speech_samples / total_samples)


# Global VAD instance
vad: Optional[SileroVAD] = None

# =============================================================================
# Context Manager (NEW in v2.0)
# =============================================================================

class ContextManager:
    """
    Manages transcription context for Whisper prompt conditioning.
    
    Handles:
    - Context accumulation and trimming
    - Intelligent reset based on silence and punctuation
    - Repetition detection to prevent hallucination loops
    """
    
    def __init__(self):
        self.context: str = ""
        self.recent_segments: deque = deque(maxlen=config.repetition_window)
        self.last_text_time: float = time.time()
        self.context_enabled: bool = config.use_context
        
        # Statistics
        self.context_resets: int = 0
        self.repetitions_detected: int = 0
    
    def get_context(self) -> Optional[str]:
        """
        Get the current context to send as Whisper prompt.
        Returns None if context is disabled or empty.
        """
        if not self.context_enabled or not self.context:
            return None
        
        return self._trim_to_token_limit(self.context)
    
    def _trim_to_token_limit(self, text: str) -> str:
        """
        Trim text to approximately max_context_tokens.
        Tries to respect sentence boundaries.
        """
        # Rough estimation: 1 token ≈ 0.75 words for most languages
        words = text.split()
        max_words = int(config.max_context_tokens * 0.75)
        
        if len(words) <= max_words:
            return text
        
        # Take last N words
        truncated = ' '.join(words[-max_words:])
        
        # Try to start at a sentence boundary
        # Look for sentence-ending punctuation followed by space
        for punct in config.sentence_end_punctuation:
            pattern = f'\\{punct}\\s+'
            matches = list(re.finditer(pattern, truncated))
            if matches:
                # Start after the first sentence boundary
                start_pos = matches[0].end()
                if start_pos < len(truncated) * 0.5:  # Don't trim more than half
                    return truncated[start_pos:]
        
        return truncated
    
    def update(self, new_text: str, silence_duration: float) -> Tuple[bool, str]:
        """
        Update context with new transcription.
        
        Args:
            new_text: The new transcription segment
            silence_duration: Silence duration before this segment (seconds)
            
        Returns:
            Tuple of (context_was_reset, reason)
        """
        self.last_text_time = time.time()
        reset_reason = ""
        
        # Check if we should reset context
        should_reset, reset_reason = self._should_reset_context(silence_duration)
        
        if should_reset:
            self.context = ""
            self.recent_segments.clear()
            self.context_resets += 1
            logger.info(f"Context reset: {reset_reason}")
        
        # Check for repetition
        if config.detect_repetition and self._is_repetition(new_text):
            self.repetitions_detected += 1
            logger.warning(f"Repetition detected, resetting context: '{new_text[:50]}...'")
            self.context = ""
            self.recent_segments.clear()
            return True, "repetition_detected"
        
        # Add new text to context
        if new_text.strip():
            if self.context:
                self.context += " " + new_text.strip()
            else:
                self.context = new_text.strip()
            
            self.recent_segments.append(new_text.strip().lower())
        
        return should_reset, reset_reason
    
    def _should_reset_context(self, silence_duration: float) -> Tuple[bool, str]:
        """
        Determine if context should be reset based on silence and punctuation.
        """
        # Long silence = definite reset
        if silence_duration >= config.context_reset_silence:
            return True, f"long_silence_{silence_duration:.1f}s"
        
        # Check for sentence-final punctuation + moderate silence
        if self.context and silence_duration >= config.context_reset_punctuation_silence:
            last_char = self.context.rstrip()[-1] if self.context.rstrip() else ""
            if last_char in config.sentence_end_punctuation:
                return True, f"sentence_end_plus_silence_{silence_duration:.1f}s"
        
        return False, ""
    
    def _is_repetition(self, new_text: str) -> bool:
        """
        Check if new_text is a repetition of recent segments.
        Uses simple word overlap ratio.
        """
        if not self.recent_segments:
            return False
        
        new_words = set(new_text.strip().lower().split())
        if not new_words:
            return False
        
        for recent in self.recent_segments:
            recent_words = set(recent.split())
            if not recent_words:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(new_words & recent_words)
            union = len(new_words | recent_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity >= config.repetition_threshold:
                    return True
        
        return False
    
    def reset(self):
        """Manually reset the context."""
        self.context = ""
        self.recent_segments.clear()
        self.context_resets += 1
        logger.info("Context manually reset")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable context management."""
        self.context_enabled = enabled
        if not enabled:
            self.context = ""
        logger.info(f"Context management {'enabled' if enabled else 'disabled'}")
    
    def get_stats(self) -> Dict:
        """Get context manager statistics."""
        return {
            "context_enabled": self.context_enabled,
            "context_length": len(self.context),
            "context_words": len(self.context.split()) if self.context else 0,
            "context_resets": self.context_resets,
            "repetitions_detected": self.repetitions_detected,
            "recent_segments_count": len(self.recent_segments)
        }


# =============================================================================
# Text Reconciler (NEW in v2.0.4)
# =============================================================================

class TextReconciler:
    """
    Reconciles overlapping transcriptions to remove duplicate text.
    
    When audio chunks overlap, the transcription of the overlapping region
    appears in both chunks. This class detects and removes the duplicate
    text from the beginning of the new transcription.
    """
    
    def __init__(self):
        self.previous_text: str = ""
        self.reconciliation_count: int = 0
        self.words_removed: int = 0
    
    def reconcile(self, new_text: str, had_overlap: bool) -> Tuple[str, bool]:
        """
        Reconcile new transcription with previous one.
        
        Args:
            new_text: The new transcription text
            had_overlap: Whether the audio chunk had overlap from previous chunk
            
        Returns:
            Tuple of (reconciled_text, was_reconciled)
        """
        if not had_overlap or not self.previous_text or not new_text:
            self.previous_text = new_text
            return new_text, False
        
        reconciled, was_reconciled = self._remove_overlap(new_text)
        
        if was_reconciled:
            self.reconciliation_count += 1
        
        self.previous_text = new_text  # Store original for next comparison
        return reconciled, was_reconciled

    def reconcile_by_timestamps(
        self,
        words: List[Dict],
        overlap_duration: float
    ) -> Tuple[str, bool]:
        """
        Timestamp-based overlap reconciliation (NEW in v2.2, updated v2.3.0).

        Discards any word whose *start* time falls within the effective overlap
        window [0, overlap_duration + tolerance), keeping everything from the
        first word that starts at or after that boundary.

        v2.3.0 changes:
          - Boundary extended by config.overlap_timestamp_tolerance (default
            0.15s) to absorb Whisper's ±150ms word-timing variance.
          - Hybrid fallback: if timestamp path removes 0 words, the text-based
            path is tried as a safety net for edge-case duplicates.
        """
        if not words:
            return "", False

        # v2.3.0: extend boundary by tolerance to absorb timing variance.
        # A tiny epsilon (1e-9) guards against floating-point rounding when
        # overlap_duration + tolerance lands just above a word's start time.
        effective_boundary = overlap_duration + config.overlap_timestamp_tolerance - 1e-9

        new_words = [w for w in words if w["start"] >= effective_boundary]
        discarded  = len(words) - len(new_words)

        if discarded > 0:
            reconciled_text = " ".join(w["word"] for w in new_words).strip()
            self.reconciliation_count += 1
            self.words_removed += discarded
            logger.debug(
                f"Timestamp reconciliation: discarded {discarded} word(s) "
                f"(start < {effective_boundary:.2f}s), kept {len(new_words)}"
            )
            full_text = " ".join(w["word"] for w in words).strip()
            self.previous_text = full_text
            return reconciled_text, True

        # v2.3.0 hybrid fallback: timestamp path found nothing —
        # try text-based as a safety net for words whose timestamps are
        # slightly above the boundary despite being acoustic duplicates.
        full_text = " ".join(w["word"] for w in words).strip()
        if self.previous_text:
            text_reconciled, was_text_reconciled = self._remove_overlap(full_text)
            if was_text_reconciled:
                logger.debug(
                    "Timestamp reconciliation found nothing; "
                    "text-based fallback applied"
                )
                self.previous_text = full_text
                return text_reconciled, True

        # Nothing to remove
        self.previous_text = full_text
        return full_text, False
    
    def _remove_overlap(self, new_text: str) -> Tuple[str, bool]:
        """
        Remove overlapping text from the beginning of new_text.

        Finds the longest suffix of previous text that matches a prefix of new text,
        using fuzzy word matching to handle:
          - Case differences        ("och" / "Och")
          - ASR truncations         ("et"  / "e", "well" / "wel")
          - Minor character errors  ("gesot" / "geso")

        Minimum match: 2 words for fuzzy/partial matches; 1 word allowed when
        the matched word is >= 3 characters (avoids false positives on "a", "e" etc.).
        """
        prev_words = self.previous_text.strip().split()
        new_words  = new_text.strip().split()

        if not prev_words or not new_words:
            return new_text, False

        check_words = config.overlap_reconcile_words
        prev_tail   = prev_words[-check_words:] if len(prev_words) >= check_words else prev_words

        best_match_len = 0
        best_match_quality = 0.0   # average per-word similarity of the best match

        for suffix_len in range(len(prev_tail), 0, -1):
            suffix = prev_tail[-suffix_len:]

            if suffix_len > len(new_words):
                continue

            total_sim      = 0.0
            all_matched    = True
            confirmed_so_far = 0  # number of words already confirmed in this suffix

            for i, prev_word in enumerate(suffix):
                matched, sim = self._words_match(
                    prev_word, new_words[i],
                    context_confirmed=confirmed_so_far
                )
                if not matched:
                    all_matched = False
                    break
                total_sim      += sim
                confirmed_so_far += 1

            if all_matched:
                avg_quality = total_sim / suffix_len

                # Minimum thresholds:
                #   ≥2 words  → always accept
                #   1 word    → only accept if the normalised word is ≥3 chars
                if suffix_len >= 2:
                    best_match_len     = suffix_len
                    best_match_quality = avg_quality
                    break
                elif suffix_len == 1:
                    norm = self._normalize(prev_tail[-1])
                    if len(norm) >= 3:
                        best_match_len     = 1
                        best_match_quality = avg_quality
                    break

        if best_match_len >= 1:
            remaining_words = new_words[best_match_len:]
            if remaining_words:
                self.words_removed += best_match_len
                reconciled = " ".join(remaining_words)
                logger.debug(
                    f"Reconciled overlap: removed {best_match_len} word(s) "
                    f"(avg similarity {best_match_quality:.2f})"
                )
                return reconciled, True
            else:
                logger.debug(
                    f"Reconciled overlap: entire chunk was overlap ({best_match_len} words)"
                )
                return "", True

        return new_text, False

    @staticmethod
    def _normalize(word: str) -> str:
        """Lowercase and strip punctuation from a word."""
        return word.lower().strip(".,!?;:'\"()-–—")

    @staticmethod
    def _static_words_match(word1: str, word2: str) -> Tuple[bool, float]:
        """
        Static wrapper for word matching — callable without a TextReconciler
        instance (used by _apply_last_word_withholding first-word guard).
        """
        n1 = TextReconciler._normalize(word1)
        n2 = TextReconciler._normalize(word2)
        if n1 == n2:
            return True, 1.0
        shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
        sim = len(shorter) / len(longer) if longer else 0.0
        if len(shorter) >= 2 and longer.startswith(shorter) and sim >= 0.5:
            return True, sim
        if len(n1) >= 3 and len(n2) >= 3:
            ratio = SequenceMatcher(None, n1, n2).ratio()
            if ratio >= 0.75:
                return True, ratio
        return False, 0.0

    def _words_match(
        self, word1: str, word2: str, context_confirmed: int = 0
    ) -> Tuple[bool, float]:
        """
        Check if two words match, returning (matched: bool, similarity: float).

        context_confirmed: how many prior words in the current suffix have
        already matched exactly. When > 0 we are inside a confirmed run, so
        we can afford to be slightly more lenient on short words without
        increasing the global false-positive risk.

        Handles:
          1. Exact match after case/punctuation normalisation  ("och"  / "Och")
          2. Prefix/truncation match for partial ASR words     ("et"   / "e",
                                                                "well" / "wel")
             — in a confirmed multi-word context a 1-char prefix is allowed;
               otherwise requires the shorter form to be ≥2 chars.
          3. Fuzzy character similarity via SequenceMatcher    ("gesot"/ "geso")
             — only for words ≥3 chars, threshold 0.75
        """
        n1 = self._normalize(word1)
        n2 = self._normalize(word2)

        # 1. Exact match
        if n1 == n2:
            return True, 1.0

        # 2. Prefix / truncation match
        shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
        sim = len(shorter) / len(longer) if longer else 0.0
        # Minimum prefix length: 1 char when already inside a confirmed run,
        # 2 chars otherwise — prevents lone "e" triggering without context.
        min_prefix_len = 1 if context_confirmed > 0 else 2
        if len(shorter) >= min_prefix_len and longer.startswith(shorter) and sim >= 0.5:
            return True, sim

        # 3. Fuzzy character similarity (handles minor ASR variations)
        if len(n1) >= 3 and len(n2) >= 3:
            ratio = SequenceMatcher(None, n1, n2).ratio()
            if ratio >= 0.75:
                return True, ratio

        return False, 0.0
    
    def reset(self):
        """Reset the reconciler state."""
        self.previous_text = ""
    
    def get_stats(self) -> Dict:
        """Get reconciler statistics."""
        return {
            "reconciliation_count": self.reconciliation_count,
            "words_removed": self.words_removed
        }


# =============================================================================
# Post-Processor (NEW in v2.2.1)
# =============================================================================

class PostProcessor:
    """
    Language-aware text post-processing applied to each emitted segment
    before it is sent to the client.

    Rules implemented:
      1. Elision fix   — "d' Word" → "d'Word", "l' Word" → "l'Word"
                         Applies to Luxembourgish / French elided articles and
                         prepositions followed by a space before a capitalised
                         or regular word.

      2. Chunk-boundary full-stop removal — a lone "." (or "." followed only by
                         whitespace/lowercase continuation) at the very start of
                         a segment strongly suggests Whisper added a sentence-
                         ending period to the *previous* chunk's last word and
                         the current ASR pass started a new decode.  We strip
                         leading/trailing isolated punctuation that sits between
                         two words which would otherwise form a fluent phrase.

         More precisely: a full-stop is removed when it
           (a) appears at the END of a word that is immediately followed by
               a lower-case word (not a new sentence), or
           (b) appears at the START of a word (artefact like ".gesot").

         We deliberately do NOT remove full-stops before an upper-case word
         (genuine sentence boundary) or at the true end of the emitted text
         (may be a real sentence end).
    """

    # Elision tokens: article/preposition + apostrophe, space, next word
    # Covers Luxembourgish (d', l', s', n', m', w') and French overlap
    ELISION_RE = re.compile(
        r"\b([dDlLsSnNmMwW]')(\s+)(\S)",   # e.g. "d' R" → "d'R"
        re.UNICODE
    )

    # Spurious full-stop patterns:
    #   (A) word-ending dot before lowercase:  "oder. wéi" → "oder wéi"
    #   (B) dot-prefixed word:                 ".gesot"    → "gesot"
    TRAILING_DOT_BEFORE_LOWER_RE = re.compile(
        r"(\w)\.\s+([a-zàáâäæçèéêëîïôöùûüÿœßëäöüàâçéèêëîïôùûœæ])",
        re.UNICODE
    )
    LEADING_DOT_RE = re.compile(r"(?<!\w)\.(\w)", re.UNICODE)

    def process(self, text: str) -> str:
        if not text:
            return text

        # Rule 1: elision — remove space between article and following word
        text = self.ELISION_RE.sub(r"\1\3", text)

        # Rule 2a: trailing dot before lower-case continuation
        # Replace "word. next" with "word next" (keep the space)
        text = self.TRAILING_DOT_BEFORE_LOWER_RE.sub(r"\1 \2", text)

        # Rule 2b: leading dot artefact ".word" → "word"
        text = self.LEADING_DOT_RE.sub(r"\1", text)

        return text.strip()


# =============================================================================
# Repetition Classifier (NEW in v2.3.0)
# =============================================================================

class RepetitionClassifier:
    """
    Distinguishes chunk-boundary artefact repetitions from genuine speech
    repetitions using a 3-signal majority vote.

    Signals
    -------
    1. Position   — artefact always at word index 0-1 of the new chunk.
    2. Temporal   — artefact has near-zero gap between the two occurrences
                    (same acoustic event re-appearing from the overlap audio).
    3. Context    — artefact has no new preceding context in the new chunk;
                    genuine repetition has a different preceding word.

    At least 2-of-3 signals must agree before the word is removed.
    This preserves real repetitions such as:
      "nee, nee"  /  "jo, jo"  /  "zwou Wochen, dräi Wochen"

    Usage
    -----
    Call ``classify_first_word`` after overlap reconciliation, passing the
    remaining word list (with timestamps), the accumulated text so far, the
    absolute end-time of the last emitted word, and the overlap duration.
    """

    def classify_first_word(
        self,
        new_words: List[Dict],
        accumulated_text: str,
        last_emitted_word_end: float,
        overlap_duration: float,
    ) -> Tuple[bool, str]:
        """
        Check whether the first word of *new_words* is a boundary artefact.

        Parameters
        ----------
        new_words             : word dicts from ASR (with 'word', 'start', 'end')
        accumulated_text      : full transcript emitted so far
        last_emitted_word_end : absolute 'end' time of the last word that was
                                sent to the client (set to 0.0 if unknown)
        overlap_duration      : configured overlap window in seconds

        Returns
        -------
        (is_artefact: bool, reason: str)
        """
        if not new_words or not accumulated_text:
            return False, "no_context"

        first      = new_words[0]
        first_norm = TextReconciler._normalize(first["word"])

        if not first_norm:
            return False, "empty_word"

        # Is this word present in the recent tail of accumulated text?
        prev_words = accumulated_text.strip().split()
        recent_tail = [TextReconciler._normalize(w) for w in prev_words[-8:]]
        if first_norm not in recent_tail:
            return False, "word_not_in_recent_context"

        # ── Signal 1: Position ────────────────────────────────────────────────
        # Artefacts appear at the very start of the new chunk (index 0).
        # This signal always fires True here (we only call classify_first_word
        # for the first word), but keeps the voting structure symmetric.
        position_artefact = True

        # ── Signal 2: Temporal gap ────────────────────────────────────────────
        # How far past the overlap boundary does this word start?
        # gap = first["start"] - overlap_dur
        #   ≈ 0   → word is right at the boundary → likely same acoustic event
        #   > 0.2 → word is well into new audio   → likely genuine new occurrence
        # We do NOT use last_emitted_word_end here because the absolute timing
        # reference is the overlap boundary itself, not the previous chunk's end.
        gap = first["start"] - overlap_duration
        temporal_artefact = gap < config.repetition_genuine_gap_threshold

        # ── Signal 3: Context ─────────────────────────────────────────────────
        # A genuine repetition has its OWN preceding word in the new chunk
        # (e.g., "dräi Wochen" → "dräi" precedes "Wochen", which is new context).
        # An artefact at position 0 has no preceding word at all.
        if len(new_words) >= 2:
            second_norm     = TextReconciler._normalize(new_words[1]["word"])
            # If the second word also appears in the recent tail, both are likely
            # artefacts continuing from the overlap.
            context_artefact = second_norm in recent_tail
        else:
            # Only one word in the chunk — no following context → artefact signal
            context_artefact = True

        # ── Vote ──────────────────────────────────────────────────────────────
        votes       = sum([position_artefact, temporal_artefact, context_artefact])
        is_artefact = votes >= 2

        reason = (
            f"votes={votes}/3 "
            f"[pos={'A' if position_artefact else 'G'}, "
            f"gap={gap:.3f}s={'A' if temporal_artefact else 'G'}, "
            f"ctx={'A' if context_artefact else 'G'}]"
        )
        logger.debug(
            f"RepetitionClassifier '{first['word']}': "
            f"{'ARTEFACT' if is_artefact else 'GENUINE'} — {reason}"
        )
        return is_artefact, reason


# =============================================================================
# Audio Buffer Manager
# =============================================================================

class AudioBufferManager:
    """Manages audio buffering with VAD-based chunking and overlap support"""
    
    def __init__(self, vad_instance: SileroVAD):
        self.vad = vad_instance
        self.buffer = np.array([], dtype=np.int16)
        self.buffer_start_time = time.time()
        self.last_speech_time = time.time()
        self.last_send_time = time.time()
        self.silence_duration = 0.0
        
        # Track silence duration at chunk send time (for context management)
        self.chunk_silence_duration = 0.0
        
        # Track send reason for logging
        self.last_send_reason = ""
        
        # NEW in v2.0.4: Overlap buffer
        self.overlap_buffer = np.array([], dtype=np.int16)
        self.overlap_samples = int(config.chunk_overlap_duration * config.sample_rate)
        self._last_chunk_had_overlap = False

        # Per-session chunk parameters (NEW in v2.2.1)
        # Initialised from global config; can be overridden per session via
        # TranscriptionSession.set_chunk_params() without touching global state.
        self.silence_threshold     : float = config.silence_threshold
        self.max_chunk_duration    : float = config.max_chunk_duration
        self.periodic_send_interval: float = config.periodic_send_interval
        
        # NEW in v2.0.5: Track when periodic interval was first reached
        # Used to implement "wait for pause" with timeout
        self.periodic_interval_reached_time: Optional[float] = None
    
    def add_audio(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Add audio data and return chunk if ready to send"""
        new_audio = np.frombuffer(audio_data, dtype=np.int16)
        self.buffer = np.concatenate([self.buffer, new_audio])
        
        buffer_duration = len(self.buffer) / config.sample_rate
        time_since_last_send = time.time() - self.last_send_time
        
        # Check for speech in recent audio
        recent_samples = int(0.5 * config.sample_rate)
        if len(self.buffer) >= recent_samples:
            recent_audio = self.buffer[-recent_samples:]
            has_speech = self.vad.has_speech(recent_audio, config.sample_rate)
            
            if has_speech:
                self.last_speech_time = time.time()
                self.silence_duration = 0.0
            else:
                self.silence_duration = time.time() - self.last_speech_time
        
        # Determine if we should send
        should_send = False
        send_reason = ""
        
        # Condition 1: Max duration reached (always send, can't wait forever)
        if buffer_duration >= self.max_chunk_duration:
            should_send = True
            send_reason = "max_duration"
            self.periodic_interval_reached_time = None  # Reset
        
        # Condition 2: Silence detected after minimum duration (natural pause)
        elif (buffer_duration >= config.min_chunk_duration and 
              self.silence_duration >= self.silence_threshold):
            should_send = True
            send_reason = "silence_detected"
            self.periodic_interval_reached_time = None  # Reset
        
        # Condition 3: VAD-aware periodic send (NEW in v2.0.5)
        elif (self.periodic_send_interval > 0 and
              buffer_duration >= config.min_chunk_duration and
              time_since_last_send >= self.periodic_send_interval):
            
            # Check if we have enough silence for a natural pause
            has_natural_pause = self.silence_duration >= config.periodic_min_silence
            
            if has_natural_pause:
                # Good - we have a natural pause, send now
                should_send = True
                send_reason = "periodic_pause"
                self.periodic_interval_reached_time = None
            else:
                # No pause yet - start waiting or check timeout
                if self.periodic_interval_reached_time is None:
                    # First time reaching interval, start waiting
                    self.periodic_interval_reached_time = time.time()
                    logger.debug(f"Periodic interval reached, waiting for natural pause...")
                
                # Check if we've waited too long
                wait_time = time.time() - self.periodic_interval_reached_time
                if wait_time >= config.periodic_max_wait:
                    # Timeout - send anyway (overlap will help)
                    should_send = True
                    send_reason = "periodic_timeout"
                    self.periodic_interval_reached_time = None
                    logger.debug(f"Periodic timeout after {wait_time:.1f}s, sending with overlap")
        
        if should_send:
            # v2.0.5: Prepend overlap from previous chunk
            if config.enable_chunk_overlap and len(self.overlap_buffer) > 0:
                chunk = np.concatenate([self.overlap_buffer, self.buffer])
                self._last_chunk_had_overlap = True
            else:
                chunk = self.buffer.copy()
                self._last_chunk_had_overlap = False
            
            # Store overlap for next chunk (last N samples of current buffer)
            if config.enable_chunk_overlap and len(self.buffer) >= self.overlap_samples:
                self.overlap_buffer = self.buffer[-self.overlap_samples:].copy()
            else:
                self.overlap_buffer = np.array([], dtype=np.int16)
            
            self.chunk_silence_duration = self.silence_duration
            self.last_send_reason = send_reason
            self.reset()
            return chunk
        
        return None
    
    def reset(self):
        """Reset the buffer after sending a chunk (preserves overlap buffer)"""
        self.buffer = np.array([], dtype=np.int16)
        self.buffer_start_time = time.time()
        self.last_send_time = time.time()
        self.silence_duration = 0.0
    
    def reset_for_new_recording(self):
        """
        Full reset for starting a new recording.
        Resets all timing state including last_speech_time and overlap.
        """
        self.buffer = np.array([], dtype=np.int16)
        self.overlap_buffer = np.array([], dtype=np.int16)  # v2.0.5: Clear overlap
        self.buffer_start_time = time.time()
        self.last_speech_time = time.time()
        self.last_send_time = time.time()
        self.silence_duration = 0.0
        self.chunk_silence_duration = 0.0
        self.last_send_reason = ""
        self._last_chunk_had_overlap = False
        self.periodic_interval_reached_time = None  # v2.0.5: Reset periodic wait state
    
    def get_remaining(self) -> Optional[np.ndarray]:
        """Get remaining audio in buffer"""
        if len(self.buffer) > int(0.5 * config.sample_rate):
            # v2.0.5: Include overlap for final chunk too
            if config.enable_chunk_overlap and len(self.overlap_buffer) > 0:
                chunk = np.concatenate([self.overlap_buffer, self.buffer])
                self._last_chunk_had_overlap = True
            else:
                chunk = self.buffer.copy()
                self._last_chunk_had_overlap = False
            
            self.chunk_silence_duration = self.silence_duration
            self.last_send_reason = "finalize"
            self.overlap_buffer = np.array([], dtype=np.int16)  # Clear on finalize
            self.reset()
            return chunk
        return None
    
    def get_chunk_silence_duration(self) -> float:
        """Get the silence duration at the time of last chunk."""
        return self.chunk_silence_duration
    
    def get_send_reason(self) -> str:
        """Get the reason for the last send."""
        return self.last_send_reason
    
    def had_overlap(self) -> bool:
        """Check if the last returned chunk had overlap prepended."""
        return self._last_chunk_had_overlap


# =============================================================================
# ASR API Client (Async)
# =============================================================================

class ASRClient:
    """Async client for the ASR REST API"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=config.asr_timeout)
    
    async def transcribe(
        self,
        audio: np.ndarray,
        language: str = "lb",
        diarization: str = "Disabled",
        output_format: str = "text",
        prompt: Optional[str] = None,
        word_timestamps: bool = False   # NEW in v2.2: request per-word timestamps
    ) -> Dict:
        """
        Send audio to the ASR API.
        
        Args:
            audio: Audio samples as numpy array
            language: Language code
            diarization: Diarization setting
            output_format: Output format
            prompt: Optional context/prompt for Whisper (NEW in v2.0)
            word_timestamps: Request word-level timestamps (NEW in v2.2).
                             When True, forces outfmt=json and adds
                             word_timestamps=true to the request.
        """
        start_time = time.time()
        audio_duration = len(audio) / config.sample_rate
        
        try:
            # Convert to WAV
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(config.channels)
                wav_file.setsampwidth(config.sample_width)
                wav_file.setframerate(config.sample_rate)
                wav_file.writeframes(audio.tobytes())
            
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
            
            # Build parameters - optimized for real-time (v2.0)
            # When word_timestamps requested, force outfmt=json (v2.2)
            effective_format = "json" if word_timestamps else output_format
            params = {
                "language": language,
                "diarization": diarization,
                "outfmt": effective_format,
                # Real-time optimizations
                "vad_filter": str(config.asr_vad_filter).lower(),  # Gateway handles VAD
                "beam_size": str(config.asr_beam_size),            # Lower = faster
            }
            
            # Request word-level timestamps when needed (v2.2)
            if word_timestamps:
                params["word_timestamps"] = "true"
            
            # Add prompt if provided (NEW in v2.0)
            if prompt:
                params["prompt"] = prompt
                logger.debug(f"Sending with prompt: '{prompt[:50]}...' ({len(prompt)} chars)")
            
            logger.info(f"Sending {audio_duration:.1f}s audio to ASR API" + 
                       (f" with {len(prompt)} char prompt" if prompt else "") +
                       (" [word timestamps]" if word_timestamps else ""))
            
            # Async HTTP POST
            response = await self.client.post(
                config.asr_api_url,
                files={"audio_file": ("audio.wav", wav_bytes, "audio/wav")},
                params=params
            )
            
            elapsed = time.time() - start_time
            rtf = elapsed / audio_duration if audio_duration > 0 else 0
            
            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    result = response.text.strip()

                # ── Parse response ──────────────────────────────────────────
                text = ""
                words = []   # list of {"word": str, "start": float, "end": float}

                if word_timestamps and isinstance(result, dict):
                    # JSON response with word timestamps
                    # faster-whisper / WhisperX style: result["segments"][*]["words"]
                    text = result.get("text", "").strip()
                    for seg in result.get("segments", []):
                        for w in seg.get("words", []):
                            words.append({
                                "word":  w.get("word", "").strip(),
                                "start": float(w.get("start", 0.0)),
                                "end":   float(w.get("end",   0.0)),
                            })
                    # Fallback: top-level "words" key
                    if not words:
                        for w in result.get("words", []):
                            words.append({
                                "word":  w.get("word", "").strip(),
                                "start": float(w.get("start", 0.0)),
                                "end":   float(w.get("end",   0.0)),
                            })
                elif isinstance(result, str):
                    text = result.strip()

                elif isinstance(result, list):
                    # Production ASR API returns a diarisation list even when
                    # diarization=Disabled — extract text and words from segments.
                    # [{"speaker": "SPEAKER_00", "text": "...", "words": [...]}]
                    text = " ".join(
                        seg.get("text", "").strip()
                        for seg in result
                        if seg.get("text", "").strip()
                    )
                    if word_timestamps:
                        for seg in result:
                            for w in seg.get("words", []):
                                word_text = w.get("word", "").strip()
                                if word_text:
                                    words.append({
                                        "word":  word_text,
                                        "start": float(w.get("start", 0.0)),
                                        "end":   float(w.get("end",   0.0)),
                                    })

                elif isinstance(result, dict):
                    text = result.get("text", str(result)).strip()
                else:
                    text = str(result).strip()

                # Clean surrounding quotes
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                
                logger.info(f"Transcription: {elapsed:.2f}s (RTF: {rtf:.2f})"
                            + (f" [{len(words)} words with timestamps]" if words else ""))
                
                return {
                    "success": True,
                    "text": text,
                    "words": words,          # NEW in v2.2: may be empty list
                    "language": language,
                    "audio_duration": audio_duration,
                    "processing_time": elapsed,
                    "rtf": rtf,
                    "used_prompt": prompt is not None
                }
            else:
                logger.error(f"ASR API error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "text": ""
                }
                
        except httpx.TimeoutException:
            logger.error("ASR API timeout")
            return {"success": False, "error": "Request timeout", "text": ""}
        except httpx.ConnectError as e:
            logger.error(f"ASR API connection error: {e}")
            return {"success": False, "error": "Connection failed", "text": ""}
        except Exception as e:
            logger.error(f"ASR API error: {e}")
            return {"success": False, "error": str(e), "text": ""}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# =============================================================================
# Translation Client (NEW in v2.1.0)
# =============================================================================

# Thread pool for running synchronous translation function
_translation_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="translation")


class TranslationClient:
    """
    Client for LocalMT translation service using localmt_translate function.
    
    Uses ThreadPoolExecutor to run the synchronous Ollama-based translation
    without blocking the async event loop.
    """
    
    def __init__(self):
        pass  # No HTTP client needed - using local function
    
    async def translate(
        self,
        text: str,
        source_lang: str = "lb",
        target_lang: str = "en"
    ) -> Dict:
        """
        Translate text using local localmt_translate function.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dict with translation result
        """
        if not text or not text.strip():
            return {
                "success": True,
                "translation": "",
                "source_lang": source_lang,
                "target_lang": target_lang
            }
        
        start_time = time.time()
        
        try:
            logger.info(f"Translating {len(text)} chars: {source_lang} -> {target_lang}")
            
            # Run synchronous translation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            translation = await loop.run_in_executor(
                _translation_executor,
                localmt_translate,
                text,
                source_lang,
                target_lang
            )
            
            elapsed = time.time() - start_time
            
            # Handle result
            if translation is None:
                logger.error("Translation returned None")
                return {
                    "success": False,
                    "error": "Translation returned None",
                    "translation": ""
                }
            
            # Clean up translation result
            translation = str(translation).strip()
            
            # Clean quotes if present
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
            
            logger.info(f"Translation completed: {elapsed:.2f}s - '{translation[:50]}...'")
            
            return {
                "success": True,
                "translation": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "processing_time": elapsed
            }
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {"success": False, "error": str(e), "translation": ""}
    
    async def close(self):
        """Cleanup (no resources to close for local function)"""
        pass


# =============================================================================
# Transcription Session
# =============================================================================

class TranscriptionSession:
    """Manages a single WebSocket transcription session"""
    
    def __init__(self, session_id: str, vad_instance: SileroVAD):
        self.session_id = session_id
        self.vad = vad_instance  # Store VAD for speech validation
        self.buffer_manager = AudioBufferManager(vad_instance)
        self.asr_client = ASRClient()
        self.translation_client = TranslationClient()  # NEW in v2.1.0
        self.context_manager = ContextManager()  # NEW in v2.0
        self.text_reconciler = TextReconciler()  # NEW in v2.0.4
        self.post_processor  = PostProcessor()   # NEW in v2.2.1
        self.repetition_classifier = RepetitionClassifier()  # NEW in v2.3.0
        
        self.language = config.default_language
        self.diarization = config.diarization
        self.output_format = config.output_format
        self.accumulated_text = ""
        self.segment_count = 0
        
        # NEW in v2.1.0: Translation settings
        self.translation_enabled = False
        self.translation_target_lang = "en"
        self.accumulated_translation = ""
        self.total_translation_time = 0.0
        
        # Statistics
        self.start_time = time.time()
        self.total_audio_seconds = 0.0
        self.total_processing_time = 0.0
        self.skipped_chunks = 0  # NEW in v2.0.2: Track skipped chunks
        
        # Pending chunk state (v2.0.3)
        self._pending_chunk = None
        self._pending_silence_duration = 0.0
        self._pending_send_reason = ""
        self._pending_had_overlap = False  # NEW in v2.0.4

        # Last-word withholding state (NEW in v2.2.0)
        # The final word of the previous chunk, held back pending confirmation.
        self._withheld_word: str = ""

        # Absolute end-time of the last word sent to the client (NEW in v2.3.0).
        # Used by RepetitionClassifier to compute the temporal gap between the
        # last emitted word and the first word of the next chunk.
        self._last_emitted_word_end: float = 0.0
    
    def _validate_audio_for_transcription(self, audio: np.ndarray) -> Tuple[bool, str]:
        """
        Validate that audio chunk contains speech worth transcribing.
        
        Returns:
            Tuple of (is_valid, reason) where reason explains why invalid
        """
        if not config.skip_empty_chunks:
            return True, ""
        
        # Check 1: Audio energy (RMS)
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32)
        else:
            audio_float = audio
        
        rms_energy = np.sqrt(np.mean(audio_float ** 2))
        
        if rms_energy < config.min_audio_energy:
            return False, f"low_energy_{rms_energy:.1f}"
        
        # Check 2: Speech ratio using VAD
        speech_ratio = self.vad.get_speech_ratio(audio, config.sample_rate)
        
        if speech_ratio < config.min_speech_ratio:
            return False, f"low_speech_ratio_{speech_ratio:.2f}"
        
        return True, ""
    
    async def process_audio(self, audio_data: bytes) -> Optional[Dict]:
        """
        Process incoming audio data.
        Returns chunk info if a chunk is ready for transcription.
        """
        self.total_audio_seconds += len(audio_data) / (config.sample_rate * config.sample_width)
        
        chunk = self.buffer_manager.add_audio(audio_data)
        if chunk is not None:
            # Store chunk for transcription
            self._pending_chunk = chunk
            self._pending_silence_duration = self.buffer_manager.get_chunk_silence_duration()
            self._pending_send_reason = self.buffer_manager.get_send_reason()
            self._pending_had_overlap = self.buffer_manager.had_overlap()  # v2.0.5
            
            # Return processing info (for visual feedback)
            audio_duration = len(chunk) / config.sample_rate
            return {
                "type": "processing",
                "audio_duration": audio_duration,
                "send_reason": self._pending_send_reason,
                "has_overlap": self._pending_had_overlap  # v2.0.5
            }
        return None
    
    async def transcribe_pending_chunk(self) -> Optional[Dict]:
        """
        Transcribe the pending chunk (if any).
        Call this after process_audio returns a 'processing' message.
        """
        if not hasattr(self, '_pending_chunk') or self._pending_chunk is None:
            return None
        
        chunk = self._pending_chunk
        silence_duration = self._pending_silence_duration
        send_reason = self._pending_send_reason
        had_overlap = self._pending_had_overlap  # v2.0.5
        
        # Clear pending
        self._pending_chunk = None
        
        return await self._transcribe_chunk(chunk, silence_duration, send_reason, had_overlap)
    
    # =========================================================================
    # Last-Word Withholding (NEW in v2.2.0)
    # =========================================================================

    def _apply_last_word_withholding(
        self, text: str, is_final: bool = False
    ) -> Tuple[str, str]:
        """
        Apply last-word withholding to a post-reconciliation text string.

        Prepends the word withheld from the *previous* chunk to the current
        emission, then withholds the *last* word of the current chunk for the
        next call (unless this is the final chunk).

        IMPORTANT: if `text` is empty (reconciliation removed the entire chunk
        because it was all overlap), we do NOT emit the withheld word yet —
        no new content has arrived to confirm the chunk boundary is stable.
        We keep holding and wait for the next chunk.

        Args:
            text:     Post-reconciliation text for this chunk.
            is_final: When True (finalize path) nothing is withheld — all text
                      is emitted so the transcript is fully flushed.

        Returns:
            (emit_text, new_withheld)
            emit_text     — text to actually send to the client and accumulate.
            new_withheld  — word to store in self._withheld_word for next chunk.
        """
        words = text.split() if text else []
        prev_withheld = self._withheld_word

        if is_final:
            # Flush everything unconditionally
            emit_parts = ([prev_withheld] if prev_withheld else []) + words
            return " ".join(emit_parts).strip(), ""

        if not words:
            # Reconciliation produced empty text — entire chunk was overlap.
            # No confirming content has arrived; keep the withheld word pending.
            return "", prev_withheld   # ← hold, do not emit

        if len(words) >= 2:
            body         = words[:-1]   # all but last word
            new_withheld = words[-1]    # last word — held back

            # v2.3.0: First-word guard ────────────────────────────────────────
            # If the first word of the new chunk fuzzy-matches the withheld word
            # from the previous chunk, it is almost certainly the same acoustic
            # event appearing twice (boundary artefact).  Drop the new copy and
            # keep the withheld version (which has more prior audio context).
            if prev_withheld and body:
                matched, _ = TextReconciler._static_words_match(prev_withheld, body[0])
                if matched:
                    logger.debug(
                        f"First-word guard: '{body[0]}' matches withheld "
                        f"'{prev_withheld}' — dropping duplicate"
                    )
                    body = body[1:]  # remove the duplicate
            # ──────────────────────────────────────────────────────────────────
        else:
            # Single new word: emit it (arrival of new chunk confirms boundary),
            # nothing new to hold.
            body         = words
            new_withheld = ""

        # Emission = previous withheld word (now confirmed) + this body
        emit_parts = ([prev_withheld] if prev_withheld else []) + body
        emit_text  = " ".join(emit_parts).strip()

        return emit_text, new_withheld

    def _flush_withheld_word(self) -> Optional[Dict]:
        """
        Emit the withheld word as a minimal transcription segment.

        Called by finalize() when there is no remaining audio but a word is
        still being held back — i.e. the speaker stopped right after the last
        processed chunk.
        """
        word = self._withheld_word
        if not word:
            return None

        self._withheld_word = ""

        # Append to accumulated text
        if self.accumulated_text:
            self.accumulated_text += " " + word
        else:
            self.accumulated_text = word

        logger.debug(f"Flushing withheld word: '{word}'")

        return {
            "type": "transcription",
            "text": word,
            "accumulated_text": self.accumulated_text,
            "is_final": True,
            "language": self.language,
            "segment": self.segment_count,
            "send_reason": "withheld_flush",
            "metrics": {
                "audio_duration": 0,
                "processing_time": 0,
                "rtf": 0,
                "silence_before": 0
            },
            "context": {
                "used_prompt": False,
                "context_reset": False,
                "reset_reason": "",
                "context_length": len(self.context_manager.context)
            },
            "overlap": {
                "had_overlap": False,
                "was_reconciled": False
            }
        }

    async def _transcribe_chunk(
        self, 
        audio: np.ndarray, 
        silence_duration: float,
        send_reason: str = "",
        had_overlap: bool = False,  # NEW in v2.0.5
        is_final: bool = False      # NEW in v2.2.0: True on finalize path
    ) -> Optional[Dict]:
        """Transcribe an audio chunk"""
        
        # NEW in v2.0.3: Validate audio before sending to API
        is_valid, skip_reason = self._validate_audio_for_transcription(audio)
        
        if not is_valid:
            self.skipped_chunks += 1
            audio_duration = len(audio) / config.sample_rate
            logger.debug(f"Skipping chunk ({audio_duration:.1f}s): {skip_reason}")
            # v2.0.3: Return None - don't send anything to client for empty chunks
            return None
        
        self.segment_count += 1
        
        # Get context for Whisper prompt (NEW in v2.0)
        context_prompt = self.context_manager.get_context()
        
        result = await self.asr_client.transcribe(
            audio,
            language=self.language,
            diarization=self.diarization,
            output_format=self.output_format,
            prompt=context_prompt,
            # v2.2: Request word timestamps when overlap is present so we can
            # use the more accurate timestamp-based reconciliation path.
            word_timestamps=(had_overlap and config.use_word_timestamps)
        )
        
        if result["success"] and result["text"]:
            self.total_processing_time += result.get("processing_time", 0)
            
            text = result["text"].strip()
            original_text = text  # Keep original for logging
            
            # v2.2: Overlap reconciliation — prefer timestamp path, fall back to text
            was_reconciled = False
            if config.enable_chunk_overlap and had_overlap:
                words = result.get("words", [])
                if words and config.use_word_timestamps:
                    # Timestamp path (v2.3.0: with tolerance + hybrid fallback)
                    text, was_reconciled = self.text_reconciler.reconcile_by_timestamps(
                        words, config.chunk_overlap_duration
                    )
                    # v2.3.0: RepetitionClassifier — second-pass check on the
                    # first remaining word to catch any residual boundary artefact
                    # that slipped past the timestamp + text reconcilers.
                    if text and words and config.enable_repetition_classifier:
                        remaining_words = [
                            w for w in words
                            if w["word"].strip() and w["word"].strip() in text.split()
                        ]
                        if remaining_words:
                            is_artefact, reason = self.repetition_classifier.classify_first_word(
                                remaining_words,
                                self.accumulated_text,
                                self._last_emitted_word_end,
                                config.chunk_overlap_duration,
                            )
                            if is_artefact:
                                text_words = text.split()
                                if len(text_words) > 1:
                                    logger.debug(
                                        f"RepetitionClassifier removed artefact "
                                        f"'{text_words[0]}': {reason}"
                                    )
                                    text = " ".join(text_words[1:])
                else:
                    # Text-based fallback (word timestamps unavailable)
                    text, was_reconciled = self.text_reconciler.reconcile(text, had_overlap)
                if was_reconciled:
                    logger.debug(f"Reconciled: '{original_text[:50]}' -> '{text[:50]}'")
            else:
                # No overlap this chunk — keep previous_text current so the
                # NEXT overlapping chunk reconciles against the correct tail.
                self.text_reconciler.previous_text = text
            
            # Update context manager with full text (including last word).
            # Context always gets the complete transcription — it's used as a
            # Whisper prompt, so having the full word is a better hint even if
            # the last word turns out to be partially decoded.
            context_reset, reset_reason = self.context_manager.update(text, silence_duration)

            # v2.2.0: Last-word withholding ─────────────────────────────────
            # Withhold the last word from client emission; prepend the word
            # withheld from the previous chunk (now considered confirmed).
            if config.enable_last_word_withholding:
                emit_text, new_withheld = self._apply_last_word_withholding(
                    text, is_final=is_final
                )
                self._withheld_word = new_withheld
                if new_withheld:
                    logger.debug(f"Withholding last word: '{new_withheld}'")
            else:
                emit_text = text
            # ─────────────────────────────────────────────────────────────────

            # If reconciliation removed everything and no withheld word is
            # ready to emit, there is nothing to send to the client.
            if not emit_text:
                return None

            # v2.2.1: Post-processing (elision fix + spurious full-stop removal)
            emit_text = self.post_processor.process(emit_text)
            # ─────────────────────────────────────────────────────────────────

            if emit_text:
                if self.accumulated_text:
                    self.accumulated_text += " " + emit_text
                else:
                    self.accumulated_text = emit_text

            # v2.3.0: record the end time of the last word we just emitted so
            # RepetitionClassifier can compute temporal gaps for the next chunk.
            result_words = result.get("words", [])
            if result_words:
                self._last_emitted_word_end = float(result_words[-1].get("end", 0.0))
            
            # NEW in v2.1.0: Translation — translate emit_text (what client sees)
            translation_result = None
            if self.translation_enabled and emit_text:
                translation_result = await self.translation_client.translate(
                    text=emit_text,
                    source_lang=self.language,
                    target_lang=self.translation_target_lang
                )
                
                if translation_result["success"]:
                    self.total_translation_time += translation_result.get("processing_time", 0)
                    translation_text = translation_result.get("translation", "")
                    
                    if translation_text:
                        if self.accumulated_translation:
                            self.accumulated_translation += " " + translation_text
                        else:
                            self.accumulated_translation = translation_text
            
            response = {
                "type": "transcription",
                "text": emit_text,           # client-visible text (withheld word excluded)
                "accumulated_text": self.accumulated_text,
                "is_final": True,
                "language": self.language,
                "segment": self.segment_count,
                "send_reason": send_reason,
                "metrics": {
                    "audio_duration": result.get("audio_duration", 0),
                    "processing_time": result.get("processing_time", 0),
                    "rtf": result.get("rtf", 0),
                    "silence_before": silence_duration
                },
                # NEW in v2.0: Context info
                "context": {
                    "used_prompt": result.get("used_prompt", False),
                    "context_reset": context_reset,
                    "reset_reason": reset_reason,
                    "context_length": len(self.context_manager.context)
                },
                # NEW in v2.0.4: Overlap info
                "overlap": {
                    "had_overlap": had_overlap,
                    "was_reconciled": was_reconciled
                },
                # NEW in v2.2.0: Withholding info
                "withholding": {
                    "withheld_word": self._withheld_word,
                    "enabled": config.enable_last_word_withholding
                }
            }
            
            # Add translation info if enabled (NEW in v2.1.0)
            if self.translation_enabled:
                response["translation"] = {
                    "enabled": True,
                    "target_lang": self.translation_target_lang,
                    "text": translation_result.get("translation", "") if translation_result else "",
                    "accumulated_translation": self.accumulated_translation,
                    "success": translation_result.get("success", False) if translation_result else False,
                    "processing_time": translation_result.get("processing_time", 0) if translation_result else 0
                }
            
            return response
        elif not result["success"]:
            return {
                "type": "error",
                "error": result.get("error", "Unknown error"),
                "segment": self.segment_count
            }
        
        return None
    
    async def finalize(self) -> Optional[Dict]:
        """
        Process remaining audio and flush any withheld word.

        v2.2.0: passes is_final=True so _transcribe_chunk emits the withheld
        word together with any remaining text.  If there is no remaining audio
        but a word is still being held, _flush_withheld_word emits it as a
        minimal segment.
        """
        remaining = self.buffer_manager.get_remaining()
        if remaining is not None:
            silence_duration = self.buffer_manager.get_chunk_silence_duration()
            send_reason      = self.buffer_manager.get_send_reason()
            had_overlap      = self.buffer_manager.had_overlap()
            return await self._transcribe_chunk(
                remaining, silence_duration, send_reason,
                had_overlap, is_final=True           # v2.2.0
            )

        # No remaining audio — flush any withheld word on its own
        return self._flush_withheld_word()
    
    def set_language(self, language: str) -> bool:
        """Update session language"""
        if language in config.supported_languages:
            self.language = language
            # Reset context when language changes (NEW in v2.0)
            self.context_manager.reset()
            logger.info(f"Session {self.session_id}: language set to {language}")
            return True
        return False
    
    def clear(self):
        """Clear accumulated text and context"""
        self.accumulated_text = ""
        self.accumulated_translation = ""  # NEW in v2.1.0
        self.segment_count = 0
        self.context_manager.reset()  # NEW in v2.0
        self._withheld_word = ""      # NEW in v2.2.0
        self._last_emitted_word_end = 0.0  # NEW in v2.3.0
    
    def set_context_enabled(self, enabled: bool):
        """Enable or disable context management (NEW in v2.0)"""
        self.context_manager.set_enabled(enabled)

    def set_chunk_params(
        self,
        silence_threshold: Optional[float] = None,
        max_chunk_duration: Optional[float] = None,
        periodic_send_interval: Optional[float] = None,
    ) -> dict:
        """
        Update per-session audio chunking parameters (NEW in v2.2.1).

        All arguments are optional — only those provided are changed.
        Values are clamped to safe ranges to prevent accidental abuse.

        Returns a dict of the values actually applied.
        """
        applied = {}

        if silence_threshold is not None:
            val = max(0.2, min(float(silence_threshold), 5.0))
            self.buffer_manager.silence_threshold = val
            applied["silence_threshold"] = val

        if max_chunk_duration is not None:
            val = max(2.0, min(float(max_chunk_duration), 60.0))
            self.buffer_manager.max_chunk_duration = val
            applied["max_chunk_duration"] = val

        if periodic_send_interval is not None:
            val = max(0.0, min(float(periodic_send_interval), 60.0))
            self.buffer_manager.periodic_send_interval = val
            applied["periodic_send_interval"] = val

        if applied:
            logger.info(
                f"Session {self.session_id}: chunk params updated — {applied}"
            )
        return applied
    
    def reset_context(self):
        """Manually reset context (NEW in v2.0)"""
        self.context_manager.reset()
    
    # =========================================================================
    # Translation Methods (NEW in v2.1.0)
    # =========================================================================
    
    def set_translation_enabled(self, enabled: bool):
        """Enable or disable translation"""
        self.translation_enabled = enabled
        logger.info(f"Session {self.session_id}: translation {'enabled' if enabled else 'disabled'}")
    
    def set_translation_target_lang(self, target_lang: str) -> bool:
        """Set translation target language"""
        if target_lang in config.translation_target_languages:
            self.translation_target_lang = target_lang
            logger.info(f"Session {self.session_id}: translation target set to {target_lang}")
            return True
        return False
    
    def reset_for_new_recording(self):
        """
        Reset session state for a new recording without closing the connection.
        Called when user stops recording but wants to start again.
        (NEW in v2.0.3)
        """
        # Reset buffer manager with full state reset
        self.buffer_manager.reset_for_new_recording()
        
        # Clear pending chunk and related state
        self._pending_chunk = None
        self._pending_silence_duration = 0.0
        self._pending_send_reason = ""
        self._pending_had_overlap = False  # v2.0.4
        
        # Reset context for fresh start
        self.context_manager.reset()
        
        # Reset text reconciler (v2.0.4)
        self.text_reconciler.reset()

        # Reset last-word withholding state (v2.2.0)
        self._withheld_word = ""

        # Reset repetition classifier timing state (v2.3.0)
        self._last_emitted_word_end = 0.0

        # Note: We don't reset accumulated_text, accumulated_translation,
        # segment_count, or statistics because those should persist across 
        # recordings in the same session
        
        logger.info(f"Session {self.session_id}: reset for new recording")
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        elapsed = time.time() - self.start_time
        stats = {
            "session_id": self.session_id,
            "duration": elapsed,
            "total_audio_seconds": self.total_audio_seconds,
            "total_processing_time": self.total_processing_time,
            "segment_count": self.segment_count,
            "skipped_chunks": self.skipped_chunks,
            "language": self.language,
            "average_rtf": (self.total_processing_time / self.total_audio_seconds 
                          if self.total_audio_seconds > 0 else 0)
        }
        # Add context stats (NEW in v2.0)
        stats["context"] = self.context_manager.get_stats()
        # Add overlap/reconciliation stats (NEW in v2.0.4)
        stats["overlap"] = self.text_reconciler.get_stats()
        # Add withholding stats (NEW in v2.2.0)
        stats["withholding"] = {
            "enabled":       config.enable_last_word_withholding,
            "pending_word":  self._withheld_word,
        }
        # Add translation stats (NEW in v2.1.0)
        stats["translation"] = {
            "enabled": self.translation_enabled,
            "target_lang": self.translation_target_lang,
            "total_translation_time": self.total_translation_time
        }
        return stats
    
    async def close(self):
        """Cleanup resources"""
        await self.asr_client.close()
        await self.translation_client.close()  # NEW in v2.1.0


# =============================================================================
# Session Manager
# =============================================================================

class SessionManager:
    """Manages active WebSocket sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, TranscriptionSession] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, session_id: str) -> TranscriptionSession:
        """Create a new session"""
        async with self._lock:
            session = TranscriptionSession(session_id, vad)
            self.sessions[session_id] = session
            return session
    
    async def remove_session(self, session_id: str):
        """Remove a session"""
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions.pop(session_id)
                await session.close()
    
    def get_all_stats(self) -> List[Dict]:
        """Get stats for all sessions"""
        return [s.get_stats() for s in self.sessions.values()]
    
    @property
    def count(self) -> int:
        return len(self.sessions)


# Global session manager
session_manager = SessionManager()

# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global vad
    
    # Startup
    logger.info("Starting LuxASR WebSocket Gateway v2.1.0...")
    logger.info(f"Context management: {'enabled' if config.use_context else 'disabled'}")
    logger.info(f"Max context tokens: {config.max_context_tokens}")
    logger.info(f"Context reset silence: {config.context_reset_silence}s")
    logger.info(f"ASR optimizations: vad_filter={config.asr_vad_filter}, beam_size={config.asr_beam_size}")
    logger.info(f"Speech validation: skip_empty={config.skip_empty_chunks}, min_ratio={config.min_speech_ratio}")
    logger.info(f"Chunk overlap: enabled={config.enable_chunk_overlap}, duration={config.chunk_overlap_duration}s")
    logger.info(f"VAD-aware periodic: interval={config.periodic_send_interval}s, min_silence={config.periodic_min_silence}s, max_wait={config.periodic_max_wait}s")
    logger.info("Translation: using local localmt_translate (Ollama)")
    logger.info("Loading Silero VAD model...")
    vad = SileroVAD()
    logger.info(f"VAD loaded: {vad.model is not None}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down WebSocket Gateway...")


app = FastAPI(
    title="LuxASR WebSocket Gateway",
    description="Real-time speech recognition gateway for LuxASR (v2.1.0 with real-time translation support)",
    version="2.3.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=config.static_dir), name="static")

# Templates
templates = Jinja2Templates(directory=config.templates_dir)

# =============================================================================
# HTTP Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the live transcription page"""
    return templates.TemplateResponse("luxrtasr.html", {"request": request})


@app.get("/dual", response_class=HTMLResponse)
async def dual_view(request: Request):
    """Serve the dual-view transcription page (side-by-side textboxes)"""
    return templates.TemplateResponse("luxrtasr_dual.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ws_gateway_fastapi",
        "version": "2.3.0",
        "timestamp": datetime.now().isoformat(),
        "vad_loaded": vad is not None and vad.model is not None,
        "active_sessions": session_manager.count,
        "asr_api_url": config.asr_api_url,
        "buffering": {
            "silence_threshold": config.silence_threshold,
            "min_chunk_duration": config.min_chunk_duration,
            "max_chunk_duration": config.max_chunk_duration
        },
        "periodic_sending": {
            "interval": config.periodic_send_interval,
            "min_silence": config.periodic_min_silence,
            "max_wait": config.periodic_max_wait
        },
        "context_management": {
            "enabled": config.use_context,
            "max_tokens": config.max_context_tokens,
            "reset_silence": config.context_reset_silence
        },
        "asr_optimizations": {
            "vad_filter": config.asr_vad_filter,
            "beam_size": config.asr_beam_size
        },
        "speech_validation": {
            "skip_empty_chunks": config.skip_empty_chunks,
            "min_speech_ratio": config.min_speech_ratio,
            "min_audio_energy": config.min_audio_energy
        },
        "chunk_overlap": {
            "enabled": config.enable_chunk_overlap,
            "overlap_duration": config.chunk_overlap_duration,
            "reconcile_words": config.overlap_reconcile_words
        },
        "translation": {
            "method": "local (localmt_translate via Ollama)",
            "target_languages": config.translation_target_languages
        }
    }


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "asr_api_url": config.asr_api_url,
        "sample_rate": config.sample_rate,
        "supported_languages": config.supported_languages,
        "default_language": config.default_language,
        "min_chunk_duration": config.min_chunk_duration,
        "max_chunk_duration": config.max_chunk_duration,
        "silence_threshold": config.silence_threshold,
        # NEW in v2.0: Context management
        "context_management": {
            "use_context": config.use_context,
            "max_context_tokens": config.max_context_tokens,
            "context_reset_silence": config.context_reset_silence,
            "context_reset_punctuation_silence": config.context_reset_punctuation_silence,
            "detect_repetition": config.detect_repetition,
            "repetition_threshold": config.repetition_threshold
        },
        # NEW in v2.0.1: ASR optimizations
        "asr_optimizations": {
            "vad_filter": config.asr_vad_filter,
            "beam_size": config.asr_beam_size,
            "diarization": config.diarization
        }
    }


@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    return {
        "count": session_manager.count,
        "sessions": session_manager.get_all_stats()
    }


# =============================================================================
# WebSocket Route
# =============================================================================

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for live transcription.
    
    Client sends:
    - Binary: 16-bit PCM audio samples (16kHz, mono)
    - JSON text: {"type": "config", "language": "lb"}
    - JSON text: {"type": "config", "use_context": true/false}
    - JSON text: {"type": "config", "translation_enabled": true/false, "translation_target": "en"}
    - JSON text: {"type": "config", "chunk_params": {"silence_threshold": 1.5, "max_chunk_duration": 10, "periodic_send_interval": 5}}
    - JSON text: {"type": "stop"}
    - JSON text: {"type": "clear"}
    - JSON text: {"type": "reset_context"}
    
    Server sends:
    - JSON: {"type": "connected", ...}
    - JSON: {"type": "processing", "audio_duration": 2.5, "send_reason": "silence_detected"}
    - JSON: {"type": "transcription", "text": "...", "send_reason": "...", "translation": {...}, ...}
    - JSON: {"type": "recording_stopped", ...}  (v2.0.3: confirms stop, session stays open)
    - JSON: {"type": "error", "error": "..."}
    """
    await websocket.accept()
    
    session_id = f"ws_{int(time.time() * 1000)}"
    logger.info(f"New WebSocket connection: {session_id}")
    
    session = await session_manager.create_session(session_id)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "version": "2.3.0",
            "config": {
                "sample_rate": config.sample_rate,
                "language": config.default_language,
                "supported_languages": config.supported_languages,
                # NEW in v2.0
                "context_management": {
                    "enabled": config.use_context,
                    "max_tokens": config.max_context_tokens
                },
                # NEW in v2.1.0
                "translation": {
                    "available": True,
                    "target_languages": config.translation_target_languages
                },
                # NEW in v2.2.1
                "chunk_params": {
                    "silence_threshold":      config.silence_threshold,
                    "max_chunk_duration":     config.max_chunk_duration,
                    "periodic_send_interval": config.periodic_send_interval,
                }
            }
        })
        
        while True:
            message = await websocket.receive()
            
            # Binary audio data
            if "bytes" in message:
                audio_data = message["bytes"]
                
                # Step 1: Check if chunk is ready
                processing_info = await session.process_audio(audio_data)
                
                if processing_info:
                    # Step 2: Send processing notification (visual feedback)
                    await websocket.send_json(processing_info)
                    
                    # Step 3: Transcribe the chunk
                    result = await session.transcribe_pending_chunk()
                    
                    # Step 4: Send result (if any - None means chunk was skipped)
                    if result:
                        await websocket.send_json(result)
            
            # Text message (JSON control)
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "config":
                        response = {"type": "config"}
                        
                        if "language" in data:
                            if session.set_language(data["language"]):
                                response["language"] = data["language"]
                        
                        # NEW in v2.0: Context control
                        if "use_context" in data:
                            session.set_context_enabled(data["use_context"])
                            response["use_context"] = data["use_context"]
                        
                        # NEW in v2.1.0: Translation control
                        if "translation_enabled" in data:
                            session.set_translation_enabled(data["translation_enabled"])
                            response["translation_enabled"] = data["translation_enabled"]
                        
                        if "translation_target" in data:
                            if session.set_translation_target_lang(data["translation_target"]):
                                response["translation_target"] = data["translation_target"]

                        # NEW in v2.2.1: Per-session chunk parameter control
                        if "chunk_params" in data:
                            cp = data["chunk_params"]
                            applied = session.set_chunk_params(
                                silence_threshold      = cp.get("silence_threshold"),
                                max_chunk_duration     = cp.get("max_chunk_duration"),
                                periodic_send_interval = cp.get("periodic_send_interval"),
                            )
                            if applied:
                                response["chunk_params"] = applied
                        
                        await websocket.send_json(response)
                    
                    elif msg_type == "stop":
                        # v2.0.3: Don't close connection - just finalize and reset
                        try:
                            result = await session.finalize()
                            if result:
                                await websocket.send_json(result)
                        except Exception as e:
                            logger.warning(f"Error during finalize: {e}")
                        
                        # Reset session state for next recording
                        try:
                            session.reset_for_new_recording()
                        except Exception as e:
                            logger.warning(f"Error during reset: {e}")
                        
                        await websocket.send_json({
                            "type": "recording_stopped",
                            "message": "Recording stopped, ready for new recording"
                        })
                        logger.info(f"Session {session.session_id}: stop processed, connection stays open")
                    
                    elif msg_type == "clear":
                        session.clear()
                        await websocket.send_json({"type": "cleared"})
                    
                    # NEW in v2.0: Manual context reset
                    elif msg_type == "reset_context":
                        session.reset_context()
                        await websocket.send_json({
                            "type": "context_reset",
                            "message": "Context has been reset"
                        })
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error in {session_id}: {e}")
    
    finally:
        stats = session.get_stats()
        await session_manager.remove_session(session_id)
        logger.info(f"Session {session_id} ended - {stats}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ws_gateway:app",
        host="127.0.0.1",
        port=5002,
        reload=False,
        log_level="info"
    )
