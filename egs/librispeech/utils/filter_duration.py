from lhotse import CutSet
import os

def filter_short_and_long_utterances(
    cuts: CutSet, min_duration: float, max_duration: float
) -> CutSet:
    def remove_short_and_long_utt(c):
        # Keep only utterances with duration between 0.6 second and 20 seconds
        if c.duration < min_duration or c.duration > max_duration:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True

    cuts = cuts.filter(remove_short_and_long_utt)

    return cuts

if __name__ == "__main__":
    # Input/output file paths
    input_cuts = "data/tokenized/cuts_train_vad.jsonl.gz"
    output_cuts = "data/tokenized/vad_lt_14/cuts_train.jsonl.gz"

    # Load data
    cuts = CutSet.from_jsonl(input_cuts)
    print(f"Original cuts: {len(cuts)}")

    # Filter short audio
    min_duration = 0.5
    max_duration = 14.0
    cuts = filter_short_and_long_utterances(cuts, min_duration, max_duration)
    print(f"Filtered cuts: {len(cuts)}")

    # Save results
    os.makedirs(os.path.dirname(output_cuts), exist_ok=True)
    cuts.to_file(output_cuts)

    # Print statistics
    print(f"Filtered cuts saved to {output_cuts}")
