"""
Phonemize Text.

Usage example:
    python3 bin/tokenizer.py \
        --src_dir ./data/manifests --output_dir ./data/tokenized

"""
import argparse
import logging
import os
from pathlib import Path

import torch
import torch.multiprocessing
from icefall.utils import get_executor
from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm

from belle.data import (
    TextTokenizer,
    tokenize_text,
)
from belle.utils import SymbolTable

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="dev-clean test-clean",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="libritts",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=400.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    dataset_parts = args.dataset_parts.replace("--dataset-parts", "").strip()
    if dataset_parts == "all":  # LibriSpeech
        dataset_parts = [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]
    else:
        dataset_parts = dataset_parts.replace("-p", "").strip().split(" ")

    assert len(dataset_parts) >= 1

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
    )

    text_tokenizer = None
    if args.text_extractor:
        text_tokenizer = TextTokenizer(backend=args.text_extractor)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    unique_symbols = set()
    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")

    with get_executor() as ex:
        for partition, m in manifests.items():
            logging.info(
                f"Processing partition: {partition} CUDA: {torch.cuda.is_available()}"
            )
            try:
                cut_set = CutSet.from_manifests(
                    recordings=m["recordings"],
                    supervisions=m["supervisions"],
                )
            except Exception:
                cut_set = m["cuts"]

            # TextTokenizer
            if args.text_extractor:
                if (
                    args.prefix == "baker"
                    and args.text_extractor == "labeled_pinyin"
                ):
                    for c in tqdm(cut_set):
                        phonemes = c.supervisions[0].custom["tokens"]["text"]
                        unique_symbols.update(phonemes)
                else:
                    for c in tqdm(cut_set):
                        if args.prefix == "ljspeech":
                            text = c.supervisions[0].custom["normalized_text"]
                            text = text.replace("”", '"').replace("“", '"')
                            phonemes = tokenize_text(text_tokenizer, text=text)
                        elif args.prefix == "aishell":
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            c.supervisions[0].custom = {}
                        elif args.prefix == "librispeech":
                            text = c.supervisions[0].text
                            # Convert from all uppercase to lowercase
                            text = text.lower()
                            c.supervisions[0].text = text
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            c.supervisions[0].custom = {}
                        else:
                            assert args.prefix == "libritts"
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                        c.supervisions[0].custom["tokens"] = {"text": phonemes}
                        unique_symbols.update(phonemes)

            cuts_filename = f"{args.prefix}_cuts_{partition}.{args.suffix}"
            cut_set.to_file(f"{args.output_dir}/{cuts_filename}")

    if args.text_extractor:
        unique_phonemes = SymbolTable()
        for s in sorted(list(unique_symbols)):
            unique_phonemes.add(s)
        logging.info(f"{len(unique_symbols)} unique phonemes: {unique_symbols}")

        unique_phonemes_file = f"{args.output_dir}/unique_text_tokens.k2symbols"
        unique_phonemes.to_file(unique_phonemes_file)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
