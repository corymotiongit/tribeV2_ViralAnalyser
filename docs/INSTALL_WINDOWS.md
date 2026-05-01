# Windows Installation

This app is designed as a local Windows workflow for reviewing videos with Meta TRIBE v2.

## Requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| OS | Windows 10/11 64-bit | Windows 11 64-bit |
| Python | Python 3.11 | Python 3.11 |
| RAM | 16 GB | 32 GB or more |
| CPU | Modern 4-core CPU | 8-core CPU or better |
| GPU | Not required; CPU mode is supported | NVIDIA GPU with CUDA-capable PyTorch for faster inference |
| Disk space | 15 GB free | 30 GB+ free on SSD |
| Internet | Required on first launch | Stable broadband for first model download |
| Browser | Any modern browser for the app | Google Chrome for reliable PDF export |

Fresh GitHub downloads include CPU fallback for machines without CUDA. The first launch downloads Python packages, official TRIBE v2 model files, the Whisper speech model, and local video/audio tooling.

Optional components:

- Hugging Face login, only if the official model page asks for access.
- Ollama with a local Qwen model, only for local recommendation copy rewriting.

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
