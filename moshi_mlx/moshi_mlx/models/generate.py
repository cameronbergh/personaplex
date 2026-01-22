# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Iterator, Callable, Union
import sys
import numpy as np
import mlx.core as mx
import sphn

from ..models import Lm
from ..modules.conditioner import ConditionTensor
from ..utils import sampling


SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS    = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)
AUDIO_TOKENS_PER_STREAM = 8

def create_sinewave(duration: float, sample_rate: int) -> np.ndarray:
    """Return a 440 Hz 'silent' sinewave of the given duration."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = 0.5
    return amplitude * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


def normalize_audio(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize **mono** audio to a target LUFS level."""
    import pyloudnorm as pyln
    # Ensure shape is (T,)
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav[0]

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    return pyln.normalize.loudness(wav, loudness, target_lufs)


def load_audio(
    filepath: str, sample_rate: int,
):
    """Yields audio samples in intervals of sample_interval_size"""
    sample_pcm, sample_sr = sphn.read(filepath)
    sample_pcm = sphn.resample(
        sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=sample_rate
    )  # shape: (C, T)
    return sample_pcm

def _iterate_audio(sample_pcm, sample_interval_size, max_len=sys.maxsize, pad=True):
    cnt = 0
    while sample_pcm.shape[-1] > 0 and cnt < max_len:
        sample = sample_pcm[:, :sample_interval_size]
        sample_pcm = sample_pcm[:, sample_interval_size:]
        if sample_pcm.shape[-1] == 0 and pad:
            sample = np.concatenate(
                [
                    sample,
                    np.zeros(
                        (
                            sample.shape[0],
                            sample_interval_size - sample.shape[-1],
                        )
                    ),
                ],
                axis=1,
            )
        cnt += 1
        yield sample[0:1]  # shape: (1, T)


def encode_from_mimi(mimi, samples):
    """
    Takes an iterator of samples, encodes them;
    and yields the encoded samples one sample at a time in the same order.
    """
    # Assuming mimi is a generic object that has an encode method
    # taking (B, C, T) and returning (B, K, F)
    # or it is rustymimi.StreamTokenizer

    # We will assume mimi is passed from local.py which might be rustymimi
    # rustymimi.StreamTokenizer has encode(pcm_data) and get_encoded()

    is_rustymimi = hasattr(mimi, "encode") and hasattr(mimi, "get_encoded")

    for sample in samples:
        # sample shape: (1, T) or (1, C, T) ?
        # _iterate_audio returns (1, T) if C=1.
        # Wait, _iterate_audio code: yield sample[0:1] -> (1, T).
        # rustymimi expects (T, 1) or (T,)?
        # local.py: pcm_data = in_data[:, 0] -> (1920,)

        if is_rustymimi:
            pcm = sample[0] # (T,)
            # rustymimi expects float32 array
            mimi.encode(pcm.astype(np.float32))
            while True:
                encoded = mimi.get_encoded()
                if encoded is not None:
                    # encoded: (steps, K) -> (K, steps)
                    yield encoded.transpose() # (K, steps)
                    break
                # If None, maybe it needs more data?
                # But _iterate_audio provides chunks of frame_size.
                # Assuming 1 frame in -> 1 frame out (with some latency, but mimi is streaming)
                # If latency, we might need to pump more zeros or handle it.
                # But here we just yield what we get.
                break
        else:
             # MLX model or PyTorch model?
             # Assuming MLX model for now or something compatible
             # Not implemented for generic MLX model yet as we focus on local.py usage
             pass


class LmGen:
    def __init__(
        self,
        model: Lm,
        max_steps: int,
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
        batch_size: int = 1,
        cfg_coef: float = 1.0,
        check: bool = False,
        on_text_hook=None,
        on_audio_hook=None,
        audio_silence_frame_cnt: int = 1,
        text_prompt_tokens: Optional[list[int]] = None,
        save_voice_prompt_embeddings: bool = False,
        sample_rate: int = 24000, # Default for Moshi
        frame_rate: float = 12.5,
    ):
        self.batch_size = batch_size
        self.model: Lm = model
        self.text_sampler = text_sampler
        self.audio_sampler = audio_sampler
        self.max_steps = max_steps
        self.check = check
        self.num_codebooks = 1 + model.cfg.audio_codebooks
        self.gen_sequence = mx.full(
            shape=(self.batch_size, self.num_codebooks, max_steps),
            vals=self.ungenerated_token,
            dtype=mx.int32,
        )
        self.step_idx = 0
        self.audio_padding_token = self.model.cfg.audio_padding_token
        self.audio_delays = self.model.cfg.audio_delays
        self.max_delay = max(self.audio_delays)
        self.main_codebooks = self.model.cfg.depformer.num_slices
        self.cfg_coef = cfg_coef
        self.on_text_hook = on_text_hook
        self.on_audio_hook = on_audio_hook

        self.audio_silence_frame_cnt = audio_silence_frame_cnt
        self.text_prompt_tokens = text_prompt_tokens
        self.save_voice_prompt_embeddings = save_voice_prompt_embeddings
        self._sample_rate = sample_rate
        self._frame_rate = frame_rate
        self._frame_size = int(self._sample_rate / self._frame_rate)

        self.voice_prompt = None
        self.voice_prompt_audio: Optional[np.ndarray] = None
        self.voice_prompt_cache: Optional[mx.array] = None
        self.voice_prompt_embeddings: Optional[mx.array] = None

        self.zero_text_code = 3 # PAD

    @property
    def zero_token(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    def _step(
        self,
        other_audio_tokens: mx.array,
        ct: ConditionTensor | None = None,
        cross_attention_src: mx.array | None = None,
        text_token: Union[int, mx.array, None] = None,
    ) -> tuple[mx.array, mx.array]:
        if self.step_idx >= self.max_steps:
            raise ValueError(f"reached max-steps {self.max_steps}")
        if self.step_idx == 0:
            if text_token is not None:
                text_tokens = mx.array([text_token], dtype=mx.int32).reshape(self.batch_size, 1)
            else:
                text_tokens = mx.full(
                    shape=(self.batch_size, 1),
                    vals=self.model.cfg.text_out_vocab_size,
                    dtype=mx.int32,
                )
        else:
            if text_token is not None:
                text_tokens = mx.array([text_token], dtype=mx.int32).reshape(self.batch_size, 1)
            else:
                text_tokens = self.gen_sequence[:, 0, self.step_idx - 1 : self.step_idx]

        self.gen_sequence[:, 1 + self.main_codebooks :, self.step_idx] = (
            other_audio_tokens
        )
        audio_tokens = []
        for cb_idx, delay in enumerate(self.audio_delays):
            gen_idx = self.step_idx - 1 - delay
            if gen_idx >= 0:
                audio_token = self.gen_sequence[:, cb_idx + 1, gen_idx][None]
            else:
                audio_token = mx.array([[self.audio_padding_token]])
            if (audio_token == self.ungenerated_token).any():  # type: ignore
                raise ValueError(
                    f"ungenerated value in audio tokens cb: {cb_idx} step: {self.step_idx}"
                )
            audio_tokens.append(audio_token)
        if (text_tokens == self.ungenerated_token).any():  # type: ignore
            raise ValueError(f"ungenerated value in text tokens {self.step_idx}")
        text_tokens, audio_tokens, transformer_out = self.model._sample(
            text_tokens,
            audio_tokens,
            self.text_sampler,
            self.audio_sampler,
            ct=ct,
            cross_attention_src=cross_attention_src,
            cfg_coef=self.cfg_coef,
            on_text_hook=self.on_text_hook,
            on_audio_hook=self.on_audio_hook,
        )

        assert audio_tokens is None or audio_tokens.shape[-2] == (
            self.model.cfg.generated_codebooks
        ), "invalid output audio-token shape"
        self.gen_sequence[:, 0, self.step_idx] = text_tokens.squeeze(-1)
        for cb_idx, delay in enumerate(self.audio_delays[: self.main_codebooks]):
            gen_idx = self.step_idx - delay
            if gen_idx >= 0:
                self.gen_sequence[:, cb_idx + 1, gen_idx] = audio_tokens[:, cb_idx, 0]
        self.step_idx += 1
        return text_tokens, transformer_out

    def step(
        self,
        other_audio_tokens: mx.array,
        ct: ConditionTensor | None = None,
        cross_attention_src: mx.array | None = None,
        text_token: Union[int, mx.array, None] = None,
    ) -> mx.array:
        return self._step(other_audio_tokens, ct, cross_attention_src, text_token)

    def step_with_extra_heads(
        self,
        other_audio_tokens: mx.array,
        ct: ConditionTensor | None = None,
        cross_attention_src: mx.array | None = None,
        text_token: Union[int, mx.array, None] = None,
    ) -> tuple[mx.array, list[mx.array]]:
        text, transformer_out = self._step(other_audio_tokens, ct, cross_attention_src, text_token)
        extra_heads = [
            mx.softmax(eh(transformer_out), axis=-1) for eh in self.model.extra_heads
        ]
        return text, extra_heads

    def last_audio_tokens(self) -> Optional[mx.array]:
        gen_idx = self.step_idx - 1 - self.max_delay
        if gen_idx < 0:
            return None
        tokens = self.gen_sequence[:, 1 : 1 + self.main_codebooks, gen_idx]

        if (tokens == self.audio_padding_token).any():  # type: ignore
            return None
        if (tokens == self.ungenerated_token).any():  # type: ignore
            raise ValueError(f"ungenerated value in last-audio tokens {self.step_idx}")
        return tokens

    def load_voice_prompt(self, voice_prompt: str):
        self.voice_prompt = voice_prompt
        raw_audio = load_audio(
            voice_prompt, self._sample_rate,
        )  # shape: (1, T) for mono

        # Normalize to -24 LUFS (mono-safe)
        raw_audio = normalize_audio(raw_audio, self._sample_rate, -24.0)

        # Keep shape (1, T)
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, :]

        self.voice_prompt_audio = raw_audio
        self.voice_prompt_cache = None
        self.voice_prompt_embeddings = None

    def load_voice_prompt_embeddings(self, path: str):
        # NOT IMPLEMENTED YET FOR MLX (requires format compatibility)
        # self.voice_prompt = path
        # state = torch.load(path)
        # self.voice_prompt_audio = None
        # self.voice_prompt_embeddings = state["embeddings"]
        # self.voice_prompt_cache = state["cache"]
        print("Warning: loading voice prompt embeddings not implemented for MLX")
        pass

    def _encode_zero_frame(self) -> mx.array:
        return mx.array(SILENCE_TOKENS, dtype=mx.int32).reshape(1, 8)

    def _encode_sine_frame(self) -> mx.array:
        return mx.array(SINE_TOKENS, dtype=mx.int32).reshape(1, 8)

    def _encode_voice_prompt_frames(self, mimi):
        return encode_from_mimi(
            mimi,
            _iterate_audio(
                self.voice_prompt_audio,
                sample_interval_size=self._frame_size,
                pad=True,
            ),
        )

    def _step_voice_prompt_core(self, mimi) -> Iterator[None]:
        if self.voice_prompt_embeddings is not None:
             # Not implemented
             pass
        elif self.voice_prompt_audio is not None:
            for voice_prompt_frame_tokens in self._encode_voice_prompt_frames(mimi):
                yield
                # voice_prompt_frame_tokens: (K, 1) or (K, steps)
                # We expect (1, K) or (K) for step?
                # _step expects (B, K).
                # If voice_prompt_frame_tokens is (K, 1), transpose to (1, K)

                # encode_from_mimi yields (K, 1)
                tokens = mx.array(voice_prompt_frame_tokens).transpose() # (1, K)

                # Always use zero_text_code during voice prompt
                self.step(
                    other_audio_tokens=tokens, # (1, K)
                    text_token=self.zero_text_code,
                    ct=None
                )
            yield
            print('Done loading voice prompt.')

    def _step_voice_prompt(self, mimi):
        for _ in self._step_voice_prompt_core(mimi):
            pass

    def _step_audio_silence_core(self) -> Iterator[None]:
        for _ in range(self.audio_silence_frame_cnt):
            yield
            self.step(
                other_audio_tokens=self._encode_sine_frame(),
                text_token=self.zero_text_code,
                ct=None
            )
        print('Done loading audio silence.')

    def _step_audio_silence(self):
        for _ in self._step_audio_silence_core():
            pass

    def _step_text_prompt_core(self) -> Iterator[None]:
        if self.text_prompt_tokens:
            for text_prompt_token in self.text_prompt_tokens:
                yield
                self.step(
                    other_audio_tokens=self._encode_sine_frame(),
                    text_token=text_prompt_token,
                    ct=None
                )
            print('Done loading text prompt.')

    def _step_text_prompt(self):
        for _ in self._step_text_prompt_core():
            pass

    def step_system_prompts(self, mimi):
        self._step_voice_prompt(mimi)
        self._step_audio_silence()
        self._step_text_prompt()
        self._step_audio_silence()
