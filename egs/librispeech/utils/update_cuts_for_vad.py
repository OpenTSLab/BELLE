import argparse
import concurrent.futures
import copy
import os
from functools import partial

import soundfile as sf
from lhotse import CutSet
from tqdm import tqdm


def get_audio_duration(audio_path: str):
    """Read an audio file and return (duration, num_samples)."""
    try:
        info = sf.info(audio_path)
        duration = info.duration
        num_samples = info.frames if info.frames is not None else int(duration * info.samplerate)
        return duration, num_samples
    except Exception as exc:
        print(f"Failed to read audio: {audio_path} -> {exc}")
        return None, None


def update_cut(
    cut,
    old_root: str,
    new_root: str,
    drop_missing: bool,
):
    new_cut = copy.deepcopy(cut)

    for src in new_cut.recording.sources:
        old_source = src.source
        src.source = old_source.replace(old_root, new_root)

    audio_source = new_cut.recording.sources[0].source

    duration, num_samples = get_audio_duration(audio_source)
    if duration is None:
        if drop_missing:
            return None
        return new_cut

    new_cut.duration = duration
    new_cut.recording.duration = duration
    new_cut.recording.num_samples = num_samples

    for supervision in new_cut.supervisions:
        supervision.duration = duration

    return new_cut


def process_with_progress(cut_list, update_func, num_workers):
    total = len(cut_list)
    pbar = tqdm(total=total, desc="Processing")

    def process_and_update(cut):
        result = update_func(cut)
        pbar.update(1)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_and_update, cut_list))

    pbar.close()

    updated = [cut for cut in results if cut is not None]
    print(f"Processed {total} original cuts, kept {len(updated)} cuts")
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Update jsonl.gz paths and durations for VAD audio"
    )
    parser.add_argument("--input", required=True, help="Original cuts file (.jsonl.gz)")
    parser.add_argument("--output", required=True, help="Output cuts file (.jsonl.gz)")
    parser.add_argument(
        "--old-root",
        default="download/LibriSpeech",
        help="Old audio root (to replace)",
    )
    parser.add_argument(
        "--new-root",
        default="download/new_librispeech",
        help="New audio root",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 8,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop cut if audio is missing or unreadable",
    )
    args = parser.parse_args()

    cuts = CutSet.from_jsonl(args.input)
    print(f"Loaded {len(cuts)} cuts")

    update_func = partial(
        update_cut,
        old_root=args.old_root,
        new_root=args.new_root,
        drop_missing=args.drop_missing,
    )
    cut_list = list(cuts)

    updated_cuts = process_with_progress(cut_list, update_func, args.num_workers)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    CutSet.from_cuts(updated_cuts).to_file(args.output)
    print(f"Done, wrote to: {args.output}")


if __name__ == "__main__":
    main()
