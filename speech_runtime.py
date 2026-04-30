from __future__ import annotations

import threading
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch
import whisper

from analysis_settings import get_analysis_mode_profile


DEFAULT_CACHE_DIR = Path.home() / "Downloads" / "tribe_cache"
WHISPER_CACHE_DIR = Path(os.environ.get("TRIBE_CACHE_DIR", DEFAULT_CACHE_DIR)) / "whisper"
WHISPER_MODEL_NAME = "base"


@dataclass
class SpeechWord:
    text: str
    start: float
    end: float
    probability: float


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str
    no_speech_prob: float


@dataclass
class SpeechRunResult:
    model_name: str
    device: str
    language: str
    text: str
    words: list[SpeechWord]
    segments: list[SpeechSegment]


class SpeechTranscriber:
    def __init__(self, model_name: str = WHISPER_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> Any:
        with self._lock:
            if self._model is None:
                WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._model = whisper.load_model(
                    self.model_name,
                    device=self.device,
                    download_root=str(WHISPER_CACHE_DIR),
                )
            return self._model

    def transcribe(self, video_path: str | Path, analysis_mode: str | None = None) -> SpeechRunResult:
        profile = get_analysis_mode_profile(analysis_mode)
        model = self.load()
        try:
            result = model.transcribe(
                str(Path(video_path)),
                verbose=False,
                word_timestamps=True,
                fp16=self.device == "cuda",
            )
        except RuntimeError as exc:
            message = str(exc)
            if "does not contain any stream" in message or "Failed to load audio" in message:
                return SpeechRunResult(
                    model_name=self.model_name,
                    device=self.device,
                    language="unknown",
                    text="",
                    words=[],
                    segments=[],
                )
            raise

        segments: list[SpeechSegment] = []
        words: list[SpeechWord] = []
        kept_text_parts: list[str] = []

        for segment in result.get("segments") or []:
            no_speech_prob = float(segment.get("no_speech_prob") or 0.0)
            if no_speech_prob > profile.max_segment_no_speech_probability:
                continue
            segment_words = []
            for word in segment.get("words") or []:
                token = str(word.get("word", "")).strip()
                if not token:
                    continue
                start = float(word.get("start") or 0.0)
                end = float(word.get("end") or start)
                probability = float(word.get("probability") or 0.0)
                if probability < profile.min_segment_word_probability:
                    continue
                speech_word = SpeechWord(
                    text=token,
                    start=start,
                    end=end,
                    probability=probability,
                )
                words.append(speech_word)
                segment_words.append(speech_word)

            if segment_words:
                segment_text = " ".join(word.text for word in segment_words).strip()
                kept_text_parts.append(segment_text)
                segments.append(
                    SpeechSegment(
                        start=float(segment.get("start") or 0.0),
                        end=float(segment.get("end") or 0.0),
                        text=segment_text,
                        no_speech_prob=no_speech_prob,
                    )
                )

        total_speech_seconds = sum(max(0.0, word.end - word.start) for word in words)
        average_probability = (
            sum(word.probability for word in words) / len(words) if words else 0.0
        )
        if (
            len(words) < profile.min_total_words
            or total_speech_seconds < profile.min_total_speech_seconds
            or average_probability < profile.min_average_word_probability
        ):
            words = []
            segments = []
            kept_text_parts = []

        return SpeechRunResult(
            model_name=self.model_name,
            device=self.device,
            language=str(result.get("language") or "unknown"),
            text=" ".join(kept_text_parts).strip(),
            words=words,
            segments=segments,
        )
