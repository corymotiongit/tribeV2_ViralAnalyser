# Windows Installation

This app is designed as a local Windows workflow for reviewing videos with Meta TRIBE v2.

## Requirements

- Windows 10/11
- Python 3.11
- Git
- Google Chrome, used for PDF export
- FFmpeg available in `PATH`
- Hugging Face access for the official TRIBE v2 model files
- Optional: NVIDIA GPU with a CUDA-compatible PyTorch install
- Optional: Ollama with a local Qwen model for copy rewriting

## 1. Create a virtual environment

```powershell
cd C:\path\to\tribe_review_mvp
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

Install the official TRIBE v2 package using the current instructions from the official model page:

```text
https://huggingface.co/facebook/tribev2
```

If you use CUDA, install the PyTorch build that matches your GPU and driver from the official PyTorch selector.

## 3. Configure Hugging Face

TRIBE v2 model files are downloaded from Hugging Face.

```powershell
huggingface-cli login
```

Paste a token that has access to the required model files.

## 4. Configure optional cache path

By default, the app uses:

```text
%USERPROFILE%\Downloads\tribe_cache
```

To override it:

```powershell
$env:TRIBE_CACHE_DIR = "D:\tribe_cache"
```

## 5. Start the app

```powershell
.\start_mvp.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

## 6. Optional Ollama setup

The app works without Ollama. If Ollama is available, it can be used as a local copy-rewriting layer.

Install Ollama, then pull one of the supported local models:

```powershell
ollama pull qwen3.5:9b
```

If no supported Ollama model is found, the app falls back to deterministic built-in copy.

