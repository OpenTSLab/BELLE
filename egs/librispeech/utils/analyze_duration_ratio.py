#!/usr/bin/env python3
"""
Script to analyze duration * 62.5 / len(custom.tokens.text) ratio from JSONL files.
"""

import json
import gzip
import argparse
import sys
from pathlib import Path


def analyze_jsonl_file(file_path):
    """
    Analyze a JSONL file (gzipped or not) and calculate duration * 62.5 / len(custom.tokens.text) ratio.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        tuple: (min_ratio, max_ratio, avg_ratio, count, ratios_list)
    """
    ratios = []
    
    try:
        # Determine if file is gzipped
        if file_path.endswith('.gz'):
            file_opener = gzip.open
            mode = 'rt'
        else:
            file_opener = open
            mode = 'r'
            
        with file_opener(file_path, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract duration
                    duration = data.get('duration')
                    if duration is None:
                        print(f"Warning: No 'duration' field found in line {line_num}")
                        continue
                    
                    # Extract custom.tokens.text
                    # Extract custom.tokens.text from supervisions
                    supervisions = data.get('supervisions', [])
                    if not supervisions:
                        # Skip lines without supervisions
                        continue
                    
                    supervision = supervisions[0]  # Get first supervision
                    custom = supervision.get('custom', {})
                    tokens = custom.get('tokens', {})
                    text_tokens = tokens.get('text', [])
                    
                    if not text_tokens:
                        print(f"Warning: No 'custom.tokens.text' found in line {line_num}")
                        continue
                    
                    # Calculate ratio: duration * 62.5 / len(custom.tokens.text)
                    ratio = duration * 62.5 / len(text_tokens)
                    ratios.append(ratio)
                    
                    if line_num % 1000 == 0:
                        print(f"Processed {line_num} lines...")
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None
    
    if not ratios:
        print("No valid ratios found in the file.")
        return None
    
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    avg_ratio = sum(ratios) / len(ratios)
    
    return min_ratio, max_ratio, avg_ratio, len(ratios), ratios


def main():
    parser = argparse.ArgumentParser(description='Analyze duration * 62.5 / len(custom.tokens.text) ratio from JSONL files')
    parser.add_argument('file_path', help='Path to the JSONL file (can be gzipped)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    parser.add_argument('--show-distribution', action='store_true', help='Show distribution of ratios')
    
    args = parser.parse_args()
    
    if not Path(args.file_path).exists():
        print(f"Error: File '{args.file_path}' does not exist.")
        sys.exit(1)
    
    print(f"Analyzing file: {args.file_path}")
    print("=" * 50)
    
    result = analyze_jsonl_file(args.file_path)
    
    if result is None:
        sys.exit(1)
    
    min_ratio, max_ratio, avg_ratio, count, ratios = result
    
    print(f"Total samples processed: {count}")
    print(f"Minimum ratio (duration * 62.5 / len(tokens)): {min_ratio:.6f}")
    print(f"Maximum ratio: {max_ratio:.6f}")
    print(f"Average ratio: {avg_ratio:.6f}")
    
    if args.detailed:
        print("\nDetailed Statistics:")
        print("-" * 30)
        ratios_sorted = sorted(ratios)
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            idx = int(len(ratios_sorted) * p / 100) - 1
            idx = max(0, min(idx, len(ratios_sorted) - 1))
            print(f"{p:2d}th percentile: {ratios_sorted[idx]:.6f}")
    
    if args.show_distribution:
        print("\nDistribution (10 buckets):")
        print("-" * 30)
        
        # Create histogram
        num_buckets = 10
        bucket_size = (max_ratio - min_ratio) / num_buckets
        buckets = [0] * num_buckets
        
        for ratio in ratios:
            bucket_idx = min(int((ratio - min_ratio) / bucket_size), num_buckets - 1)
            buckets[bucket_idx] += 1
        
        for i, count in enumerate(buckets):
            bucket_start = min_ratio + i * bucket_size
            bucket_end = min_ratio + (i + 1) * bucket_size
            percentage = count / len(ratios) * 100
            print(f"[{bucket_start:.3f}, {bucket_end:.3f}): {count:6d} samples ({percentage:5.1f}%)")


if __name__ == "__main__":
    main()
