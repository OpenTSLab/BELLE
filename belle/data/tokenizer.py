import re
from typing import Any, Dict, List, Optional, Pattern, Union, Tuple

import torch
import torchaudio
import torch.nn as nn
from einops import rearrange
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from ..ParallelWaveGAN.parallel_wavegan.losses import MelSpectrogram
from ..ParallelWaveGAN.parallel_wavegan.utils import load_model

class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend_name = backend
        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols


class MelTokenizer(nn.Module):
    def __init__(
        self,
        sampling_rate: int = 16000,
    ):
        super().__init__()
        path = '~/BELLE/pretrained/tts-hifigan-train/hifigan-libritts-1930000steps.pkl'
        model = load_model(path)
        assert hasattr(model, "mean"), "Feature stats are not registered."
        assert hasattr(model, "scale"), "Feature stats are not registered."
        model.remove_weight_norm()
        self.vocoder = model
        self.channels = 1
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.pad_size = int((self.n_fft - self.hop_length) / 2)
        self.sampling_rate = sampling_rate
        self.n_mels = 80
        self.mel_extractor = MelSpectrogram(
            fft_size=self.n_fft,
            fmax=7600,
            fmin=80,
            fs=self.sampling_rate,
            hop_size=self.hop_length,
            num_mels=80,
            win_length=self.win_length,
            window="hann",
        )

    def encode(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
    )-> Tuple[torch.Tensor]:
        """
        Input:
            audio: [B, 1, T] or [B, T]
            audio_lens: [B]
        Output:
            mel_spectrom: [B, T, C]
            mel_lens: [B]
        """
        if audio.ndim == 3:
            audio = rearrange(audio, 'B C T -> (B C) T')
        
        with torch.no_grad():
            spectrogram = self.mel_extractor(audio)
        
        mel_lens = torch.floor(audio_lens / self.hop_length).type(torch.int64) + 1

        if spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(0)
        mel_spectrom = rearrange(spectrogram, 'B C T -> B T C')

        return mel_spectrom, mel_lens
    
    def decode(
        self,
        mel_spectrom: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            mel_spectrom: [1, T, C] or [T, C]
        Output:
            audio: [1, 1, T]
        """
        self.vocoder.device = mel_spectrom.device
        mel_spectrom = mel_spectrom.squeeze()
        with torch.no_grad():
            audio = self.vocoder.inference(mel_spectrom, normalize_before=True)
        audio = rearrange(audio, "T C -> 1 C T")
        return audio


def tokenize_audio(tokenizer: MelTokenizer, audio_path: str, device: Any = None):
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    if sr != tokenizer.sampling_rate:
        wav = torchaudio.transforms.Resample(sr, tokenizer.sampling_rate)(wav)
    wav = wav.unsqueeze(0)

    encoded_frames, encoded_frames_len = tokenizer.encode(wav.to(device), torch.tensor([wav.shape[-1]]))
    return encoded_frames


if __name__ == "__main__":
    text_tokenizer = TextTokenizer(backend="espeak")
    text = "you'll get a book."
    phonemes = tokenize_text(text_tokenizer, text)
    print(phonemes)

    # python -m belle.data.tokenizer
