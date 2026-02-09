import sys

sys.path.append("~/index-tts")
from pathlib import Path

import torch

from tts_launch import TTSBaseLauncher
from indextts.infer import IndexTTS


class F5TTSInferenceLauncher(TTSBaseLauncher):
    @property
    def gen_sr(self):
        return 24_000

    def load_models(self) -> dict:
        model = IndexTTS(
            model_dir="~/index-tts/checkpoints",
            cfg_path="~/index-tts/checkpoints/config.yaml"
        )
        return {
            "model": model,
        }

    def inference_single_implementation(
        self,
        args,
        models: dict,
        text: str,
        prompt_speech: str | Path,
        prompt_text: str | None = None,
    ):
        model = models["model"]
        wav, sr = model.infer(prompt_speech, text)
        wav = wav.cpu().type(torch.float32)
        wav = wav[0]
        return wav

    def audio_post_processing(self, wav):
        return wav.type(torch.int16)


if __name__ == '__main__':
    launcher = F5TTSInferenceLauncher()
    launcher.run()
