# Select multiple .jsonl.gz files and merge into one .jsonl.gz using lhotse to read.
# Provide a list of files and a list of ratios; e.g., 0.5 reads the first 50% of a file.

from lhotse import CutSet
from lhotse.utils import Pathlike
from typing import List, Optional, Union


def load_cuts(
    paths: Union[Pathlike, List[Pathlike]],
    weights: Optional[List[float]] = None,
) -> CutSet:
    """
    Load cuts from multiple JSONL files and concatenate them into a single CutSet.
    Optionally, you can specify weights to control the proportion of each file's cuts in the final CutSet.
    """
    if isinstance(paths, str):
        paths = [paths]

    cut_sets = []
    for path in paths:
        cut_set = CutSet.from_jsonl(path)
        print(f"Loaded {len(cut_set)} cuts from {path}")
        cut_sets.append(cut_set)

    if weights is not None:
        assert len(cut_sets) == len(weights), "Number of cuts and weights must match."
        selected_cut_sets = []
        for cut_set, weight, path in zip(cut_sets, weights, paths):
            num = int(len(cut_set) * weight)
            selected = cut_set[:num]
            print(f"Selected {len(selected)} cuts from {path} with weight {weight}")
            selected_cut_sets.append(selected)
        cut_sets = selected_cut_sets
    else:
        print("No weights provided, using all cuts from each file.")

    return CutSet.from_cuts(cut for cut_set in cut_sets for cut in cut_set)


data_list = [
    "egs/librispeech/data/tokenized/vad_lt_14/cuts_train_clean_100.jsonl.gz",
    "egs/librispeech/data/tokenized/vad_lt_14/cuts_train_clean_360.jsonl.gz",
    # "egs/librispeech/data/tokenized/vad_lt_14/cuts_train_other_500.jsonl.gz",
]

weights = [
    1.0,
    0.55,
    # 0.315
]  # Read the first 50% of the data

concatenated_cuts = load_cuts(data_list, weights)
# Save the concatenated cuts to a new JSONL file
output_path = "egs/librispeech/data/tokenized/vad_lt_14/cuts_train_clean_218.jsonl.gz"
concatenated_cuts.to_jsonl(output_path)

print(f"Concatenated cuts saved to {output_path}")
print(f"Number of cuts in the concatenated set: {len(concatenated_cuts)}")