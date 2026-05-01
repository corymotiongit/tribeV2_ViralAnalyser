from __future__ import annotations

import pathlib
import json
import os
import subprocess
import shutil
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import tribev2.eventstransforms as tribev2_eventstransforms

from runtime_setup import ensure_local_ffmpeg_on_path


if hasattr(pathlib, "WindowsPath"):
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]

from tribev2 import TribeModel


ensure_local_ffmpeg_on_path()

MODEL_REPO = "facebook/tribev2"
DEFAULT_CACHE_DIR = Path.home() / "Downloads" / "tribe_cache"
CACHE_DIR = Path(os.environ.get("TRIBE_CACHE_DIR", DEFAULT_CACHE_DIR))
MODEL_SNAPSHOT_DIR = CACHE_DIR / "official_model_repo"


@dataclass
class TribeRunResult:
    preds: Any
    timestamps: list[float]
    device: str
    modalities: list[str]
    events_count: int
    mesh_level: str
    subject_model: str
    hemodynamic_lag_seconds: float


class TribeVideoBackend:
    def __init__(self) -> None:
        self._model: TribeModel | None = None
        self._model_dir: Path | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> TribeModel:
        with self._lock:
            if self._model is None:
                model_dir = self._resolve_official_model_dir()
                self._model = TribeModel.from_pretrained(
                    model_dir,
                    cache_folder=CACHE_DIR,
                    device=self.device,
                    config_update={
                        "data.num_workers": 0,
                        "data.batch_size": 1,
                    },
                )
            return self._model

    def _resolve_official_model_dir(self) -> Path:
        if self._model_dir is not None:
            return self._model_dir

        from huggingface_hub import snapshot_download

        MODEL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns=["config.yaml", "best.ckpt"],
            local_dir=MODEL_SNAPSHOT_DIR,
            local_dir_use_symlinks=False,
        )
        self._model_dir = Path(snapshot_path)
        return self._model_dir

    def predict_video(self, video_path: str | Path) -> TribeRunResult:
        model = self.load()
        self._ensure_uvx_on_path()
        self._ensure_official_transcript_helper()
        original_cuda_check = tribev2_eventstransforms.torch.cuda.is_available
        tribev2_eventstransforms.torch.cuda.is_available = lambda: False
        try:
            events = model.get_events_dataframe(video_path=str(Path(video_path)))
        finally:
            tribev2_eventstransforms.torch.cuda.is_available = original_cuda_check
        preds, segments = model.predict(events=events, verbose=False)
        timestamps = [self._segment_timestamp(segment) for segment in segments]
        modalities = sorted({str(item).lower() for item in events["type"].unique().tolist()})
        return TribeRunResult(
            preds=preds,
            timestamps=timestamps,
            device=self.device,
            modalities=modalities,
            events_count=len(events),
            mesh_level="fsaverage5",
            subject_model="average",
            hemodynamic_lag_seconds=5.0,
        )

    @staticmethod
    def _segment_timestamp(segment: Any) -> float:
        start = float(getattr(segment, "start", 0.0) or 0.0)
        offset = float(getattr(segment, "offset", 0.0) or 0.0)
        return start + offset

    @staticmethod
    def _ensure_uvx_on_path() -> None:
        if shutil.which("uvx"):
            return
        candidates = [
            Path.home() / "Documents" / "ComfyUI" / ".venv" / "Scripts" / "uvx.exe",
            Path.home() / ".local" / "bin" / "uvx",
        ]
        for candidate in candidates:
            if candidate.exists():
                current_path = os.environ.get("PATH", "")
                os.environ["PATH"] = str(candidate.parent) + os.pathsep + current_path
                return

    @staticmethod
    def _ensure_official_transcript_helper() -> None:
        current = tribev2_eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio
        if getattr(current, "__name__", "") == "_compatible_get_transcript_from_audio":
            return

        def _compatible_get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
            language_codes = dict(
                english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
            )
            if language not in language_codes:
                raise ValueError(f"Language {language} not supported")

            device = "cuda" if tribev2_eventstransforms.torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            batch_size = "16" if device == "cuda" else "4"

            with tempfile.TemporaryDirectory() as output_dir:
                cmd = [
                    "uvx",
                    "whisperx",
                    str(wav_filename),
                    "--model",
                    "large-v3",
                    "--language",
                    language_codes[language],
                    "--device",
                    device,
                    "--compute_type",
                    compute_type,
                    "--batch_size",
                    batch_size,
                    "--align_model",
                    "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                    "--output_dir",
                    output_dir,
                    "--output_format",
                    "json",
                ]
                cmd = [c for c in cmd if c]
                env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
                env.setdefault("PYTHONIOENCODING", "utf-8")
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=env,
                    )
                except OSError:
                    return _empty_transcript_dataframe()
                if result.returncode != 0:
                    return _empty_transcript_dataframe()

                json_path = Path(output_dir) / f"{wav_filename.stem}.json"
                if not json_path.exists():
                    return _empty_transcript_dataframe()
                transcript = json.loads(json_path.read_text(encoding="utf-8"))

            words = []
            for i, segment in enumerate(transcript.get("segments") or []):
                sentence = segment["text"].replace('"', "")
                for word in segment.get("words", []):
                    if "start" not in word:
                        continue
                    words.append(
                        {
                            "text": word["word"].replace('"', ""),
                            "start": word["start"],
                            "duration": word["end"] - word["start"],
                            "sequence_id": i,
                            "sentence": sentence,
                        }
                    )
            return pd.DataFrame(words)

        tribev2_eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
            _compatible_get_transcript_from_audio
        )


def _empty_transcript_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["text", "start", "duration", "sequence_id", "sentence"])
