import sys

sys.path.append("~/CosyVoice")
sys.path.append("~/CosyVoice/third_party/Matcha-TTS")
from pathlib import Path
import torch

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from tts_launch import TTSBaseLauncher
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


class CosyVoiceInferenceLauncher(TTSBaseLauncher):
    @property
    def gen_sr(self):
        return self.gen_sr_

    def load_models(self) -> dict:
        cosyvoice = CosyVoice2(
            '~/CosyVoice/pretrained_models/CosyVoice2-0.5B',
            load_jit=True,
            load_trt=True,
            load_vllm=True,
            fp16=True
        )
        self.gen_sr_ = cosyvoice.sample_rate
        return {'model': cosyvoice}

    def inference_single_implementation(
        self,
        args,
        models: dict,
        text: str,
        prompt_speech: str | Path,
        prompt_text: str | None = None,
    ):
        cosyvoice = models['model']
        prompt_speech = load_wav(prompt_speech, 16_000)
        outputs = cosyvoice.inference_zero_shot(
            text, prompt_text, prompt_speech, stream=False
        )
        outputs = list(outputs)
        # assert len(
        #     outputs
        # ) == 1, f"Multiple outputs found for {text}: {len(outputs)}"
        # return outputs[0]["tts_speech"][0]
        final_speech = torch.cat(
            [i['tts_speech'][0] for i in outputs], dim=0
        )
        return final_speech


if __name__ == '__main__':
    launcher = CosyVoiceInferenceLauncher()
    launcher.run()
