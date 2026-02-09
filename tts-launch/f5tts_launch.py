import sys

sys.path.append("~/F5-TTS")
from pathlib import Path

from tts_launch import TTSBaseLauncher
from infer_utils import load_models, generate_audio


class F5TTSInferenceLauncher(TTSBaseLauncher):
    @property
    def gen_sr(self):
        return 24000

    def load_models(self) -> dict:
        ema_model, vocoder = load_models(
            ckpt_file="~/F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.safetensors"
        )
        return {
            "model": ema_model,
            "vocoder": vocoder,
        }

    def inference_single_implementation(
        self,
        args,
        models: dict,
        text: str,
        prompt_speech: str | Path,
        prompt_text: str | None = None,
    ):
        model, vocoder = models["model"], models["vocoder"]
        audio_wav, _ = generate_audio(
            model,
            vocoder,
            prompt_speech,
            prompt_text,
            text,
        )
        return audio_wav


if __name__ == '__main__':
    launcher = F5TTSInferenceLauncher()
    launcher.run()
