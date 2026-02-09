import abc
from pathlib import Path
import random

import numpy as np
from lhotse import CutSet, load_manifest_lazy
import torch
import soundfile as sf
from tqdm import tqdm
from torchaudio.transforms import Resample

from multi_gpu_cli_launch import MultiGPUWorker


class TTSBaseLauncher(MultiGPUWorker):
    @property
    @abc.abstractmethod
    def gen_sr(self):
        pass

    @property
    def target_sr(self):
        return 16_000

    def add_running_args(self):
        self.parser.add_argument(
            '--recordings',
            type=str,
            required=True,
            help='Path to recordings jsonl.gz file'
        )
        # self.parser.add_argument(
        #     '--supervisions',
        #     type=str,
        #     required=True,
        #     help='Path to supervisions jsonl.gz file'
        # )
        self.parser.add_argument(
            '--output_dir',
            type=str,
            required=True,
            help='Path to output directory'
        )
        self.parser.add_argument(
            '--audio_base_dir',
            type=str,
            required=True,
            help=
            'Path to the base audio directory when using relative paths in the cuts. '
        )
        self.parser.add_argument(
            "--speakers",
            type=str,
            default=None,
            help="Path to the speaker list"
        )
        self.parser.add_argument(
            "--language",
            type=str,
            default="en",
            choices=["zh", "en"],
        )

    def load_item_list(self, args):
        # supervisions = load_manifest_lazy(args.supervisions)
        # recordings = load_manifest_lazy(args.recordings)
        # cuts = CutSet.from_manifests(
        #     recordings=recordings, supervisions=supervisions
        # )
        print(f"Loading cuts from {args.recordings}")
        cuts = CutSet.from_jsonl(
            args.recordings,
        )
        if args.speakers is not None:
            with open(args.speakers, 'r') as f:
                speakers = set(line.strip() for line in f)
            cuts = cuts.filter(
                lambda cut: cut.supervisions[0].speaker in speakers
            )
        return cuts

    @abc.abstractmethod
    def load_models(self) -> dict:
        pass

    @abc.abstractmethod
    def inference_single_implementation(
        self,
        args,
        models: dict,
        text: str,
        prompt_speech: str | Path,
        prompt_text: str,
    ):
        pass

    def inference_single_sample(
        self,
        args,
        utt_id: str,
        models: dict,
        text: str,
        prompt_speech: str | Path,
        prompt_text: str | None = None,
    ) -> torch.Tensor | np.ndarray | None:
        try:
            wav = self.inference_single_implementation(
                args, models, text, prompt_speech, prompt_text
            )
            return wav
        except Exception as e:
            print(f"Error processing {utt_id}: {e}")
            return None

    def audio_post_processing(self, wav):
        return wav

    def process_item_list(self, cuts, args):

        models = self.load_models()
        transform = Resample(self.gen_sr, self.target_sr)

        # build speaker_to_cuts dictionary
        speaker_to_cuts = {}
        for cut in tqdm(cuts, desc="Building speaker_to_cuts dictionary"):
            if len(cut.supervisions) == 0:
                continue
            speaker = cut.supervisions[0].speaker
            if speaker not in speaker_to_cuts:
                speaker_to_cuts[speaker] = []
            speaker_to_cuts[speaker].append(cut)

        print(f"{len(speaker_to_cuts)} speakers are found.")

        output_dir = Path(args.output_dir)
        audio_base_dir = Path(args.audio_base_dir)

        for cut in tqdm(cuts, desc="Synthesizing speech segments"):
            if len(cut.supervisions) == 0:
                continue

            fpath = Path(cut.recording.sources[0].source)
            if fpath.exists():  # absolute path
                output_path_r = fpath.relative_to(audio_base_dir)
            else:  # relative path
                output_path_r = fpath
            output_path = output_dir / output_path_r

            if output_path.exists():
                print(
                    f"Output file {output_path} already exists, skipping {cut.id}"
                )
                continue

            text = cut.supervisions[0].text
            speaker = cut.supervisions[0].speaker

            # select a random prompt from the same speaker
            available_cuts = [
                c for c in speaker_to_cuts[speaker] if c.id != cut.id
            ]
            if not available_cuts:
                print(
                    f"Speaker {speaker} has no other samples, skipping {cut.id}"
                )
                continue

            # filter cuts by duration (3-10 seconds preferred)
            preferred_cuts = [
                c for c in available_cuts 
                if 3.0 <= c.duration <= 10.0
            ]
            
            # use preferred cuts if available, otherwise use all available cuts
            cuts_to_choose_from = preferred_cuts if preferred_cuts else available_cuts

            num_attempts = 0

            while num_attempts < 5:
                prompt_cut = random.choice(cuts_to_choose_from)
                prompt_path = Path(prompt_cut.recording.sources[0].source)
                if not prompt_path.exists():
                    prompt_path = audio_base_dir / prompt_path
                prompt_text = prompt_cut.supervisions[0].text

                wav = self.inference_single_sample(
                    args, cut.id, models, text, prompt_path, prompt_text
                )
                if wav is not None:
                    wav = torch.as_tensor(wav).float()
                    wav = transform(wav)
                    wav = self.audio_post_processing(wav)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(output_path, wav, self.target_sr)
                    break
                else:
                    num_attempts += 1
