import sys
sys.path.append("~/Amphion")
from pathlib import Path

import torch
from torch import nn
from torchaudio.transforms import Resample

from infer import load_mask_gct


class MaskGCT(nn.Module):
    def __init__(self, local_rank=0):
        super().__init__()
        self.device = torch.device(f"cuda:{local_rank}")
        self.model = load_mask_gct()
        self.sr = 24000
        self.resample_16k = Resample(orig_freq=24000, new_freq=16000)

    def inference(
        self,
        text: str,
        prompt_speech: str | Path,
        task_key: str,
        prompt_text: str | None = None,
    ):
        try:
            wav = self.model.maskgct_inference(
                prompt_speech,
                prompt_text,
                text,
                language="en",
                target_language="en"
            )
            audio_wav = torch.as_tensor(wav).float().to(self.device)
            audio_wav_16k = self.resample_16k(audio_wav).cpu()
            return audio_wav_16k.squeeze(0), None, 16000, 0
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None, 16000, 0
