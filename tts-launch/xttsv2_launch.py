from pathlib import Path

import torch

from tts_launch import TTSBaseLauncher
from TTS.api import TTS


class XTTSv2InferenceLauncher(TTSBaseLauncher):
    @property
    def gen_sr(self):
        return self.gen_sr_

    def load_models(self) -> dict:
        tts_model = TTS(
            model_path="~/XTTS-v2",
            config_path=
            "~/XTTS-v2/config.json",
        )
        device = torch.device("cuda")
        tts_model.to(device)
        self.gen_sr_ = tts_model.synthesizer.output_sample_rate
        return {
            "model": tts_model,
        }

    def inference_single_implementation(
        self,
        args,
        models: dict,
        text: str,
        prompt_speech: str | Path,
        prompt_text: str | None = None,
    ):
        tts_model = models["model"]
        wav = tts_model.tts(
            text=text,
            speaker_wav=prompt_speech,
            language=args.language,
            speed=1.0,
        )
        return wav


if __name__ == '__main__':
    launcher = XTTSv2InferenceLauncher()
    launcher.run()
