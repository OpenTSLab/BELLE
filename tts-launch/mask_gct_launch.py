import sys

sys.path.insert(0, "~/Amphion")
from pathlib import Path

from tts_launch import TTSBaseLauncher
from infer import load_mask_gct


class MaskGCTInferenceLauncher(TTSBaseLauncher):
    @property
    def gen_sr(self):
        return 24_000

    def load_models(self) -> dict:
        model = load_mask_gct()
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
        wav = model.maskgct_inference(
            prompt_speech,
            prompt_text,
            text,
            language=args.language,
            target_language=args.language
        )
        return wav


if __name__ == '__main__':
    launcher = MaskGCTInferenceLauncher()
    launcher.run()
