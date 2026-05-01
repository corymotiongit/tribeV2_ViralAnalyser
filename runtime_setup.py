from __future__ import annotations

import os
from pathlib import Path


def ensure_local_ffmpeg_on_path() -> None:
    try:
        import imageio_ffmpeg
    except Exception:
        return

    try:
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return

    if not ffmpeg_path.exists():
        return

    current_path = os.environ.get("PATH", "")
    ffmpeg_dir = str(ffmpeg_path.parent)
    if ffmpeg_dir.lower() not in current_path.lower().split(os.pathsep):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
