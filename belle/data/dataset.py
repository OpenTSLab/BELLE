"""
modified from lhoste.dataset.speech_synthesis.py
"""

from typing import Callable, Dict, List, Sequence, Union
import warnings

import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone
import torchaudio

# Suppress torchaudio deprecation warnings about StreamingMediaDecoder
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
from torch.nn.utils.rnn import pad_sequence

from belle.data.collation import TextTokenCollater


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis(e.g. TTS) task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
            'text': str
            'text_tokens': (B x NumTextTokens) long tensor
            'text_tokens_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        text_token_collater: TextTokenCollater,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        tts_models: List[str] = None,
    ) -> None:
        super().__init__()

        self.text_token_collater = text_token_collater
        self.cut_transforms = ifnone(cut_transforms, [])

        self.tts_models = tts_models
        print(f"tts_models: {tts_models}")

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        audio, audio_lens = collate_audio(cuts)

        audio_paths = [cut.recording.sources[0].source for cut in cuts]
        if self.tts_models != [""] and self.tts_models:
            tts_audios = self.load_tts_audio(audio_paths)
        else:
            tts_audios = {}

        texts = [cut.supervisions[0].text for cut in cuts]
        text_tokens, text_tokens_lens = self.text_token_collater(
            [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
        )

        # Extract prompt information if available
        prompt_texts = []
        prompt_text_tokens_list = []
        prompt_audio_paths = []
        
        for cut in cuts:
            if "prompt" in cut.supervisions[0].custom:
                prompt = cut.supervisions[0].custom["prompt"]
                prompt_texts.append(prompt.get("text", ""))
                prompt_text_tokens_list.append(prompt.get("tokens", []))
                prompt_audio_paths.append(prompt.get("source", ""))
            else:
                # No prompt available for this cut
                prompt_texts.append("")
                prompt_text_tokens_list.append([])
                prompt_audio_paths.append("")
        
        # Tokenize prompt text
        prompt_text_tokens, prompt_text_tokens_lens = self.text_token_collater(prompt_text_tokens_list)
        
        # Load prompt audio
        prompt_audio_list = []
        prompt_audio_lens_list = []
        for prompt_audio_path in prompt_audio_paths:
            if prompt_audio_path:
                try:
                    prompt_audio_data, _ = torchaudio.load(prompt_audio_path)
                    prompt_audio_data = prompt_audio_data.squeeze()
                    prompt_audio_list.append(prompt_audio_data)
                    prompt_audio_lens_list.append(prompt_audio_data.size(0))
                except Exception as e:
                    print(f"Warning: Failed to load prompt audio {prompt_audio_path}: {e}")
                    # Use empty tensor as fallback
                    prompt_audio_list.append(torch.zeros(16000))
                    prompt_audio_lens_list.append(16000)
            else:
                # No prompt audio for this cut
                # print("No prompt audio path provided, using zero tensor as fallback.")
                prompt_audio_list.append(torch.zeros(16000))
                prompt_audio_lens_list.append(16000)
        
        prompt_audio = pad_sequence(prompt_audio_list, batch_first=True)
        prompt_audio_lens = torch.tensor(prompt_audio_lens_list, dtype=torch.int32)

        return {
            "utt_id": [cut.id for cut in cuts],
            "text": texts,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
            "audio": audio,
            "audio_lens": audio_lens,
            "prompt_text": prompt_texts,
            "prompt_text_tokens": prompt_text_tokens,
            "prompt_text_tokens_lens": prompt_text_tokens_lens,
            "prompt_audio": prompt_audio,
            "prompt_audio_lens": prompt_audio_lens,
            **tts_audios,
        }

    def load_tts_audio(self, audio_paths: list[str]) -> dict[str, torch.Tensor]:
        """
        Load audio files from the given paths.
        """
        tts_audios = {}
        for tts_name in self.tts_models:
            tts_audios[f"audio_{tts_name}"] = []
            tts_audios[f"audio_{tts_name}_lens"] = []
            for audio_path in audio_paths:
                assert "new_librispeech" in audio_path, f"audio_path: {audio_path}"
                tts_audio_path = audio_path.replace("new_librispeech", f"librispeech_{tts_name}_vad")
                try:
                    tts_audio = torchaudio.load(tts_audio_path)[0].squeeze()
                    tts_audios[f"audio_{tts_name}"].append(tts_audio)
                    tts_audios[f"audio_{tts_name}_lens"].append(tts_audio.size(0))
                except Exception as e:
                    # Log warning and skip corrupted/invalid audio files
                    print(f"Warning: Failed to load audio file {tts_audio_path}: {e}. Skipping this file.")
                    # Add a zero-length placeholder to maintain batch structure
                    tts_audios[f"audio_{tts_name}"].append(torch.zeros(16000))
                    tts_audios[f"audio_{tts_name}_lens"].append(16000)
            tts_audios[f"audio_{tts_name}"] = pad_sequence(tts_audios[f"audio_{tts_name}"], batch_first=True)
            tts_audios[f"audio_{tts_name}_lens"] = torch.tensor(tts_audios[f"audio_{tts_name}_lens"], dtype=torch.int32)
        return tts_audios

def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
