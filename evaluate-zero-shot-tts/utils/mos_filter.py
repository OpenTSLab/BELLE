#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
from typing import Any, Dict


def filter_by_mos_score(input_file: str, output_file: str, threshold: float) -> None:
    """
    Filter items from jsonl.gz whose mos_score is above a threshold.

    Args:
        input_file: Input jsonl.gz file path.
        output_file: Output jsonl.gz file path.
        threshold: MOS score threshold.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    filtered_count = 0
    total_count = 0
    
    print(f"Processing file: {input_file}")
    print(f"MOS score threshold: {threshold}")
    
    with gzip.open(input_file, 'rt', encoding='utf-8') as infile, \
         gzip.open(output_file, 'wt', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                total_count += 1
                
                # Check for mos_score field
                if 'mos_score' not in item:
                    print(f"Warning: line {line_num} missing mos_score, skipping")
                    continue
                
                mos_score = float(item['supervisions'][0]['custom']['mos_score'])
                
                # Keep items with score above threshold
                if mos_score >= threshold:
                    outfile.write(line + '\n')
                    filtered_count += 1
                    
                # Show progress every 10,000 lines
                if total_count % 10000 == 0:
                    print(f"Processed {total_count} lines, kept {filtered_count} records")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: line {line_num} JSON parse error: {e}")
                continue
            except (ValueError, TypeError) as e:
                print(f"Warning: line {line_num} mos_score format error: {e}")
                continue
    
    print("\nFiltering complete:")
    print(f"Total records: {total_count}")
    print(f"Kept records: {filtered_count}")
    print(f"Keep rate: {filtered_count/total_count*100:.2f}%")
    print(f"Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Filter jsonl.gz by MOS score')
    parser.add_argument('input_file', help='Input jsonl.gz file path')
    parser.add_argument('output_file', help='Output jsonl.gz file path')
    parser.add_argument('--threshold', type=float, help='MOS score threshold')
    
    args = parser.parse_args()
    
    # Check that input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: input file does not exist: {args.input_file}")
        return 1
    
    try:
        filter_by_mos_score(args.input_file, args.output_file, args.threshold)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
