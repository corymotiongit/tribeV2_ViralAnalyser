# Windows Installation

This app is designed as a local Windows workflow for reviewing videos with Meta TRIBE v2.

## Requirements

Recommended local setup:

- Windows 10/11 64-bit
- Python 3.11
- 32 GB RAM
- Modern 8-core CPU or better
- NVIDIA GPU
- 8 GB VRAM minimum, 12 GB+ preferred
- 30 GB+ free disk space, preferably on SSD
- Stable internet connection for the first launch
- Google Chrome for reliable PDF export
- Hugging Face access if the official TRIBE v2 model page asks for it
- Optional Ollama installation for local recommendation copy rewriting

The first launch downloads Python packages, official TRIBE v2 model files, the Whisper speech model, and local video/audio tooling.

## 1. Start the app

Unzip the repository archive into a normal folder, then run:

```powershell
Start_TRIBE_Review.cmd
```

The first launch creates `.venv`, installs Python dependencies, downloads the official TRIBE v2 model files, downloads the Whisper speech model, and prepares a local FFmpeg binary.

The terminal will tell you when setup is complete. After the first setup finishes, close the terminal and run `Start_TRIBE_Review.cmd` again. Later launches are much faster.

## 2. Optional Hugging Face login

If the model download fails because Hugging Face asks for access, run this inside the project folder after `.venv` exists:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

Paste a Hugging Face token that has access to the required model files, then run `Start_TRIBE_Review.cmd` again.

## 3. Optional cache path

By default, the app uses:

```text
%USERPROFILE%\Downloads\tribe_cache
```

To override it:

```powershell
$env:TRIBE_CACHE_DIR = "D:\tribe_cache"
```

## 4. Optional Ollama setup

The app works without Ollama. If Ollama is available, it can be used as a local copy-rewriting layer.

Install Ollama, then pull one of the supported local models:

```powershell
ollama pull qwen3.5:9b
```

If no supported Ollama model is found, the app falls back to deterministic built-in copy.
