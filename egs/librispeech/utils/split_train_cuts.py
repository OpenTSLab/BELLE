from lhotse import CutSet

path = "egs/librispeech/data/tokenized/lt_14/cuts_train.jsonl.gz"

train_clean_100 = []
train_clean_360 = []
train_other_500 = []

# Load data
cuts = CutSet.from_jsonl_lazy(path)

# Iterate over each cut
from tqdm import tqdm
for cut in tqdm(cuts):
    # Get original audio path
    audio_path = cut.recording.sources[0].source

    # Determine dataset type and append to the corresponding list
    if "train-clean-100" in audio_path:
        train_clean_100.append(cut)
    elif "train-clean-360" in audio_path:
        train_clean_360.append(cut)
    elif "train-other-500" in audio_path:
        train_other_500.append(cut)
    else:
        raise ValueError(f"Unknown dataset type in path: {audio_path}")

train_clean_100 = CutSet.from_cuts(train_clean_100)
print(f"train_clean_100: {len(train_clean_100)}")
train_clean_360 = CutSet.from_cuts(train_clean_360)
print(f"train_clean_360: {len(train_clean_360)}")
train_other_500 = CutSet.from_cuts(train_other_500)
print(f"train_other_500: {len(train_other_500)}")
# Save results
import os
base_dir = os.path.dirname(path)
os.makedirs(base_dir, exist_ok=True)
train_clean_100.to_file(os.path.join(base_dir, "cuts_train_clean_100.jsonl.gz"))
train_clean_360.to_file(os.path.join(base_dir, "cuts_train_clean_360.jsonl.gz"))
train_other_500.to_file(os.path.join(base_dir, "cuts_train_other_500.jsonl.gz"))