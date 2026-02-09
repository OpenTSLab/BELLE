"""
This file displays duration statistics of utterances in the manifests.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.
"""
from lhotse import load_manifest_lazy


def main():
    path = "egs/librispeech/data/tokenized/vad_lt_14/cuts_train/filter_all.jsonl.gz"
    print(f"Loading {path}...")
    cuts = load_manifest_lazy(path)
    cuts.describe()
    # speakers = set(cut.supervisions[0].speaker for cut in cuts if cut.supervisions)
    # # Count and display number of speakers
    # print(f"Number of speakers: {len(speakers)}")

if __name__ == "__main__":
    main()
