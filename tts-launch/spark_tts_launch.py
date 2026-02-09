import sys

sys.path.insert(0, "~/Spark-TTS")
from pathlib import Path

import torch

from tts_launch import TTSBaseLauncher
from cli.SparkTTS import SparkTTS


class SparkTTSInferenceLauncher(TTSBaseLauncher):
    @property
    def gen_sr(self):
        return 16_000

    def load_models(self) -> dict:
        model = SparkTTS(
            "~/Spark-TTS/pretrained_models/Spark-TTS-0.5B",
            torch.device("cuda")
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
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text=prompt_text,
        )
        return wav


if __name__ == '__main__':
    launcher = SparkTTSInferenceLauncher()
    launcher.run()
