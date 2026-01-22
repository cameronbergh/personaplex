# Moshi MLX for PersonaPlex

This is a modified version of `moshi_mlx` to support PersonaPlex features like voice cloning and system prompts on Apple Silicon (MLX).

## Installation

```bash
pip install -r requirements.txt
pip install .
```

## Usage

To run PersonaPlex on your Apple M4 Max:

```bash
python -m moshi_mlx.local \
    --hf-repo nvidia/personaplex-7b-v1 \
    --text-prompt "You enjoy having a good conversation." \
    --voice-prompt "path/to/voice_prompt.wav"
```

For quantized models (faster inference), you might need to quantize the PersonaPlex weights or use compatible ones if available.

If you don't have a voice prompt file, you can omit the `--voice-prompt` argument.
