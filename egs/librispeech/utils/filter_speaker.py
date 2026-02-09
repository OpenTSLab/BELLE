#!/usr/bin/env python3
"""
Filter cuts for a specific speaker.
"""
import gzip
import json
import os


def filter_speaker_cuts(input_path, output_path, target_speaker):
    """
    Filter cuts for a specified speaker.

    Args:
        input_path: Input jsonl.gz file path.
        output_path: Output jsonl.gz file path.
        target_speaker: Target speaker ID.
    """
    print(f"Filtering speaker={target_speaker} from {input_path}...")
    
    filtered_count = 0
    total_count = 0
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with gzip.open(input_path, 'rt', encoding='utf-8') as infile, \
         gzip.open(output_path, 'wt', encoding='utf-8') as outfile:
        
        for line in infile:
            total_count += 1
            
            # Parse JSON line
            try:
                cut_data = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON on line {total_count}")
                continue
            
            # Check speaker in supervisions
            has_target_speaker = False
            if 'supervisions' in cut_data:
                for supervision in cut_data['supervisions']:
                    if 'speaker' in supervision and supervision['speaker'] == target_speaker:
                        has_target_speaker = True
                        break
            
            # If target speaker exists, save to output
            if has_target_speaker:
                outfile.write(line)
                filtered_count += 1
                
            # Print progress every 1000 records
            if total_count % 1000 == 0:
                print(f"Processed {total_count} records, filtered {filtered_count} target records")
    
    print("Filtering completed!")
    print(f"Total processed: {total_count} records")
    print(f"Filtered: {filtered_count} records for speaker={target_speaker}")
    print(f"Results saved to: {output_path}")


def main():
    # Input file path
    input_path = "egs/librispeech/data/tokenized/cuts_train_vad.jsonl.gz"
    
    # Output file path
    output_path = "egs/librispeech/data/tokenized/speaker_39.jsonl.gz"
    
    # Target speaker
    target_speaker = "39"
    
    # Run filtering
    filter_speaker_cuts(input_path, output_path, target_speaker)


if __name__ == "__main__":
    main()
