# Troubleshooting

## The browser opens a JSON page instead of the app

Another local server may already be using port `8000`.

The launcher checks for the TRIBE app signature and will try ports `8001` through `8010` if needed. Stop the other app or open the port printed in the PowerShell window.

## PDF export fails

PDF export requires Google Chrome.

If Chrome is installed in a non-standard path, set:

```powershell
$env:TRIBE_CHROME_PATH = "C:\Path\To\chrome.exe"
```

## Hugging Face model download fails

Check that:

- you accepted any required model terms on Hugging Face;
- `huggingface-cli login` was run in the same Windows user account;
- the token has access to the required model files;
- the machine has internet access.

## `Torch not compiled with CUDA enabled`

CUDA is optional. The app should automatically run TRIBE v2 on CPU when CUDA is not available.

If this error appears, update to the latest GitHub version and run `Start_TRIBE_Review.cmd` again. Newer builds force the TRIBE feature extractors to use the same verified device as the main model.

## Video encoding is extremely slow

If a short clip shows a very long ETA during `Encoding video`, PyTorch is probably running without CUDA acceleration.

On Windows, run `Start_TRIBE_Review.cmd` again. The launcher checks for an NVIDIA GPU and installs CUDA-enabled PyTorch when needed. If CUDA still does not activate, update the NVIDIA driver and run the launcher again.

## Whisper or audio transcription fails

Check that FFmpeg is installed and available in `PATH`:

```powershell
ffmpeg -version
```

If transcription fails, the app should still continue with the visual/audio model outputs and a fallback speech state.

## TorchCodec / FFmpeg warnings

Some Python audio stacks warn when TorchCodec cannot load FFmpeg bindings. Install a compatible FFmpeg version and ensure it is visible in `PATH`.

The app also relies on FFmpeg through video/audio tooling, so fixing FFmpeg usually resolves this class of errors.

## The machine uses too much memory

Use shorter videos while testing and compare fewer variants at once. Comparison mode processes videos sequentially, but TRIBE and Whisper can still use significant memory.

For safest runs:

- test 1 video first;
- keep clips short;
- close other GPU/AI apps;
- prefer `TRIBE_CACHE_DIR` on a disk with enough space.

## Ollama is not installed

Ollama is optional. Without it, the app uses deterministic built-in wording for the recommendation layer.
