from collections import defaultdict
import random
from pathlib import Path
import json

import fire
from lhotse import CutSet, load_manifest_lazy


def count_speakers_from_dataset(
    recordings: str,
    supervisions: str,
    dataset_name: str,
    split: str | None = None
):
    recordings = load_manifest_lazy(recordings)
    supervisions = load_manifest_lazy(supervisions)
    cuts = CutSet.from_manifests(
        recordings=recordings, supervisions=supervisions
    )
    total_duration = 0
    speakers = set()
    for cut in cuts:
        total_duration += cut.duration
        speakers.add(cut.supervisions[0].speaker)
    if split is not None:
        print(
            f"Dataset {dataset_name} split {split} has {len(speakers)} speakers, "
            f"duration: {total_duration / 3600:.2f} hours."
        )
    else:
        print(
            f"Dataset {dataset_name} has {len(speakers)} speakers, "
            f"duration: {total_duration / 3600:.2f} hours."
        )


class Runner:
    def speaker_counting(self, dataset: str):
        if dataset == "vctk":
            count_speakers_from_dataset(
                "./orig_data_by_lhotse/vctk/vctk_recordings_all.jsonl.gz",
                "./orig_data_by_lhotse/vctk/vctk_supervisions_all.jsonl.gz",
                "VCTK",
            )
        elif dataset == "librispeech":
            for split in [
                "dev-clean", "dev-other", "test-clean", "test-other",
                "train-clean-100", "train-clean-360", "train-other-500"
            ]:
                count_speakers_from_dataset(
                    f"./orig_data_by_lhotse/librispeech/librispeech_recordings_{split}.jsonl.gz",
                    f"./orig_data_by_lhotse/librispeech/librispeech_supervisions_{split}.jsonl.gz",
                    dataset_name="LibriSpeech",
                    split=split
                )
        elif dataset == "libritts":
            for split in [
                "train-clean-100", "train-clean-360", "train-other-500",
                "dev-clean", "dev-other", "test-clean", "test-other"
            ]:
                count_speakers_from_dataset(
                    f"./orig_data_by_lhotse/libritts/libritts_recordings_{split}.jsonl.gz",
                    f"./orig_data_by_lhotse/libritts/libritts_supervisions_{split}.jsonl.gz",
                    dataset_name="LibriTTS",
                    split=split
                )
        elif dataset == "ljspeech":
            count_speakers_from_dataset(
                "./orig_data_by_lhotse/ljspeech/ljspeech_recordings_all.jsonl.gz",
                "./orig_data_by_lhotse/ljspeech/ljspeech_supervisions_all.jsonl.gz",
                "LJSpeech",
            )
        elif dataset == "aishell1":
            for split in ["train", "dev", "test"]:
                count_speakers_from_dataset(
                    f"./orig_data_by_lhotse/aishell1/aishell_recordings_{split}.jsonl.gz",
                    f"./orig_data_by_lhotse/aishell1/aishell_supervisions_{split}.jsonl.gz",
                    dataset_name="AISHELL-1",
                    split=split
                )
        elif dataset == "aishell3":
            for split in ["train", "test"]:
                count_speakers_from_dataset(
                    f"./orig_data_by_lhotse/aishell3/aishell3_recordings_{split}.jsonl.gz",
                    f"./orig_data_by_lhotse/aishell3/aishell3_supervisions_{split}.jsonl.gz",
                    dataset_name="AISHELL-3",
                    split=split
                )
        elif dataset == "magic_data":
            for split in ["train", "dev", "test"]:
                count_speakers_from_dataset(
                    f"./orig_data_by_lhotse/magic_data/magicdata_recordings_{split}.jsonl.gz",
                    f"./orig_data_by_lhotse/magic_data/magicdata_supervisions_{split}.jsonl.gz",
                    dataset_name="Magic Data",
                    split=split
                )

    def select_speaker_by_duration(
        self,
        recordings: str,
        supervisions: str,
        target_duration: float,  # target duration in hours
        output: str,
    ):
        target_duration_in_seconds = target_duration * 3600  # Convert hours to seconds
        recordings = load_manifest_lazy(recordings)
        supervisions = load_manifest_lazy(supervisions)
        cuts = CutSet.from_manifests(
            recordings=recordings, supervisions=supervisions
        )

        # build speaker -> cuts mapping
        speaker_to_cuts = defaultdict(list)
        for cut in cuts:
            speaker = cut.supervisions[0].speaker if cut.supervisions else None
            if speaker:
                speaker_to_cuts[speaker].append(cut)

        speakers = list(speaker_to_cuts.keys())
        random.shuffle(speakers)

        selected_speakers = []
        total_duration = 0.0

        for speaker in speakers:
            speaker_cuts = speaker_to_cuts[speaker]
            speaker_duration = sum(cut.duration for cut in speaker_cuts)

            selected_speakers.append(speaker)
            total_duration += speaker_duration

            if total_duration > target_duration_in_seconds:
                break

        print(
            f"Selected {len(selected_speakers)} speakers with total duration: "
            f"{total_duration / 3600:.2f} hours."
        )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            for speaker in selected_speakers:
                f.write(f"{speaker}\n")

    def count_dataset_utt_number(self, output: str):
        dataset_to_splits = {
            "aishell1": ["train", "dev", "test"],
            "aishell3": ["train", "test"],
            "magic_data": ["train", "dev", "test"],
            "ljspeech": ["all"],
            "vctk": ["all"],
            "libritts": ["train-clean-100", "dev-clean", "test-clean"]
        }

        with open(output, "w") as writer:
            out_data = {}
            for dataset, splits in dataset_to_splits.items():
                num_utts = 0
                for split in splits:
                    if dataset == "aishell1":
                        recordings = f"./orig_data_by_lhotse/{dataset}/aishell_recordings_{split}.jsonl.gz"
                        supervisions = f"./orig_data_by_lhotse/{dataset}/aishell_supervisions_{split}.jsonl.gz"
                    elif dataset == "magic_data":
                        recordings = f"./orig_data_by_lhotse/{dataset}/magicdata_recordings_{split}.jsonl.gz"
                        supervisions = f"./orig_data_by_lhotse/{dataset}/magicdata_supervisions_{split}.jsonl.gz"
                    else:
                        recordings = f"./orig_data_by_lhotse/{dataset}/{dataset}_recordings_{split}.jsonl.gz"
                        supervisions = f"./orig_data_by_lhotse/{dataset}/{dataset}_supervisions_{split}.jsonl.gz"
                    recordings = load_manifest_lazy(recordings)
                    supervisions = load_manifest_lazy(supervisions)
                    cuts = CutSet.from_manifests(
                        recordings=recordings, supervisions=supervisions
                    )
                    if Path(f"speakers/{dataset}/{split}.txt").exists():
                        with open(f"speakers/{dataset}/{split}.txt", "r") as f:
                            speakers = set(line.strip() for line in f)
                        cuts = cuts.filter(
                            lambda cut: cut.supervisions[0].speaker in speakers
                        )
                    num_utts += len(cuts)
                out_data[dataset] = num_utts
            json.dump(out_data, writer, indent=4)
            writer.write("\n")


if __name__ == '__main__':
    fire.Fire(Runner)
