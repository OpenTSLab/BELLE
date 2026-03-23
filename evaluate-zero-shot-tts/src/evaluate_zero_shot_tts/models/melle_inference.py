import logging
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch
from torch import nn
from icefall.utils import AttributeDict

from belle.data import (
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from belle.data.collation import get_text_token_collater
from belle.models import get_model

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=False
    )
    # assert not missing_keys
    model.to(device)
    model.eval()

    text_tokens = args.text_tokens

    return model, text_tokens, args


class MelleModel(nn.Module):
    def __init__(self, ckpt, backend, local_rank=0, dataset_name="librispeech"):
        super().__init__()
        self.text_tokenizer = TextTokenizer(backend=backend)
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
        self.model, self.text_tokens, self.args = load_model(ckpt, self.device)
        if backend == "espeak":
            self.text_tokens = os.path.join("../egs/", dataset_name, self.text_tokens)
        print('-' * 50)
        print(f"Using text tokens: {self.text_tokens}")
        print('-' * 50)
        self.text_collater = get_text_token_collater(self.text_tokens)
        self.audio_tokenizer = self.model.audio_tokenizer
        self.audio_tokenizer.to(self.device)
        self.model.eval()
        print(backend, self.text_tokens)

    def mel_recon(self, audio_prompt_mel):
        samples = self.audio_tokenizer.decode(audio_prompt_mel)
        return samples[0].cpu()

    @torch.no_grad()
    def inference(self, text, audio_file, task_key, text_prompt=None):
        if "stream_mode" in self.args and self.args.stream_mode:
            prompt_text_tokens, prompt_text_tokens_lens = self.text_collater(
                [
                    tokenize_text(
                        self.text_tokenizer, text=text_prompt.strip()
                    )
                ]
            )
            text_tokens, text_tokens_lens = self.text_collater(
                [
                    tokenize_text(
                        self.text_tokenizer, text=text.strip()
                    )
                ]
            )
        else:
            tokenized_text = tokenize_text(
                self.text_tokenizer, text=text.strip() if text_prompt is None \
                    else f"{text_prompt} {text}".strip()
            )
            text_tokens, text_tokens_lens = self.text_collater(
                [tokenized_text]
            )

        audio_prompt = tokenize_audio(self.audio_tokenizer, audio_file, self.device) # B T C

        audio_recon = self.mel_recon(audio_prompt) # 1 T

        try:
            if "stream_mode" in self.args and self.args.stream_mode:
                encoded_frames, std = self.model.inference(
                    text_tokens.to(self.device),
                    text_tokens_lens.to(self.device),
                    audio_prompt,
                    prompt_text_tokens.to(self.device),
                    prompt_text_tokens_lens.to(self.device),
                ) # 1 T C
            else:
                encoded_frames, std = self.model.inference(
                    text_tokens.to(self.device),
                    text_tokens_lens.to(self.device),
                    audio_prompt,
                ) # 1 T C
            
            if not isinstance(encoded_frames, list) and task_key == "cont":
                encoded_frames = torch.cat([audio_prompt, encoded_frames], dim=1)
            samples = self.audio_tokenizer.decode(encoded_frames)
            samples = samples[0].cpu()  # [1, T]

        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.error(f"\n-------------------\nError: {e}\n-------------------")
            return None, audio_recon, 16_000, 0

        return samples, audio_recon, 16_000, np.mean(std)
