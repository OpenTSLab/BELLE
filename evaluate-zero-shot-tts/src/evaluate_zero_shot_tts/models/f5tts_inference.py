import sys
sys.path.append("~/F5-TTS")
from pathlib import Path

import torch
from torch import nn
from torchaudio.transforms import Resample

from infer_utils import load_models, generate_audio


class F5TTSModel(nn.Module):
    def __init__(self, local_rank=0):
        super().__init__()
        self.device = torch.device(f"cuda:{local_rank}")
        self.model, self.vocoder = load_models(
            ckpt_file="~/F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.safetensors",
        )
        self.sr = 24000
        self.resample_16k = Resample(orig_freq=24000, new_freq=16000)

    def inference(
        self,
        text: str,
        prompt_speech: str | Path,
        task_key: str,
        prompt_text: str | None = None,
    ):
        assert prompt_text is not None, "F5-TTS do not support cont task"
        try:
            audio_wav, sr = generate_audio(
                self.model,
                self.vocoder,
                prompt_speech,
                prompt_text,
                text,
            )
            assert sr == self.sr
            audio_wav = torch.as_tensor(audio_wav).float().to(self.device)
            audio_wav_16k = self.resample_16k(audio_wav).cpu()
            return audio_wav_16k.squeeze(0), None, 16000, 0
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None, 16000, 0
