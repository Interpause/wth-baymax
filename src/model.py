"""File to run the audio2text model."""

import os
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, List, Literal, Union

import librosa
import soundfile
import torch
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
from typing_extensions import TypedDict
from xxhash import xxh3_64_hexdigest


class TextPart(TypedDict):
    """MsgPart."""
    type: Literal["text"]
    text: str


class AudioPart(TypedDict):
    """MsgPart."""
    type: Literal["audio"]
    audio_id: str


class Msg(TypedDict):
    """Msg."""
    role: Literal["user", "assistant", "system"]
    content: Union[List[Union[TextPart, AudioPart]], str]


Msgs = List[Msg]


class AudioCache:
    """AudioCache."""

    def __init__(self, base_path: str | bytes | os.PathLike):
        """Initialize."""
        self.base_path = Path(base_path)

    @lru_cache(maxsize=32, typed=False)
    def get(self, audio_id: str, sr: int = 16000):
        """Get audio clip."""
        path = self.base_path / f"{audio_id}.flac"
        wav, _ = librosa.load(path, sr=sr, mono=True)
        return wav

    def save(self, file: BinaryIO):
        """Save audio clip returning the audio_id."""
        wav, sr = librosa.load(file, sr=None, mono=True)
        wav.flags.writeable = False
        audio_id = xxh3_64_hexdigest(wav.data)
        if not (self.base_path / f"{audio_id}.flac").exists():
            soundfile.write(self.base_path /
                            f"{audio_id}.flac", wav, samplerate=sr)
        return audio_id


class ModelManager:
    """ModelManager."""

    def __init__(self, model_path: str, audio_cache: AudioCache):
        """Initialize."""
        self.audio_cache = audio_cache
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load the model."""
        # TODO: Pray that the quantization backend works.
        self.processor = Qwen2AudioProcessor.from_pretrained(
            self.model_path, local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, use_safetensors=True, device_map="auto")
        self.model.eval()

    @torch.inference_mode()
    def generate(self, msgs: Msgs):
        """Generate."""
        sr = self.processor.feature_extractor.sampling_rate
        audios = []
        for msg in msgs:
            if isinstance(msg["content"], list):
                for i, part in enumerate(msg["content"]):
                    if part["type"] == "audio":
                        wav = self.audio_cache.get(part["audio_id"], sr=sr)
                        audios.append(wav)
                        part["audio_url"] = f"PLACEHOLDER_{i}"

        text = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audios=audios,
                                padding=True, return_tensors="pt", sampling_rate=sr)
        inputs.input_ids = inputs.input_ids.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response
