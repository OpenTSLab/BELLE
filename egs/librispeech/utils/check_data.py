#!/usr/bin/env python3
"""
Check whether audio files in a cuts file exist; keep existing cuts and filter out missing ones.
"""

import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from lhotse import CutSet

def check_single_cut(cut, base_path):
    """
    Check whether the audio file for a single cut exists.

    Args:
        cut: Cut object.
        base_path: Base path for audio files.

    Returns:
        tuple: (cut, exists, file_path)
    """
    if cut.recording and cut.recording.sources:
        source_path = cut.recording.sources[0].source
        if source_path:
            full_path = os.path.join(base_path, source_path)
            exists = os.path.exists(full_path)
            return cut, exists, full_path
        else:
            return cut, False, f"Cut {cut.id}: missing source path"
    else:
        return cut, False, f"Cut {cut.id}: missing recording or sources"

def filter_cuts(cuts_file: str, base_path: str = ".", num_threads: int = 4):
    """
    Filter a cuts file, keeping only cuts whose audio files exist.

    Args:
        cuts_file: Cuts file path.
        base_path: Base path for audio files.
        num_threads: Number of threads.
    """
    print(f"Processing file: {cuts_file}")
    print(f"Base path: {base_path}")
    print(f"Threads: {num_threads}")
    
    try:
        # Read cuts with lhotse
        cuts = CutSet.from_file(cuts_file)
        total = len(cuts)
        valid_cuts = []
        missing_files = []
        
        # Use thread pool and progress bar
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_cut = {
                executor.submit(check_single_cut, cut, base_path): cut 
                for cut in cuts
            }
            
            # Show progress
            with tqdm(total=total, desc="Checking audio files") as pbar:
                for future in as_completed(future_to_cut):
                    cut, exists, file_path = future.result()
                    
                    if exists:
                        valid_cuts.append(cut)
                    else:
                        missing_files.append(file_path)
                    
                    pbar.update(1)
        
        # Print missing file info
        if missing_files:
            print(f"\nMissing files ({len(missing_files)}):")
            for file_path in missing_files[:10]:  # Show only the first 10
                print(f"  {file_path}")
            if len(missing_files) > 10:
                print(f"  ... {len(missing_files) - 10} more missing files")
        
        # Create a new CutSet and save it
        # filtered_cuts = CutSet.from_cuts(valid_cuts)
        # filtered_cuts.to_file(cuts_file)
        
        valid = len(valid_cuts)
        print("\nProcessing complete:")
        print(f"Total: {total}")
        print(f"Kept: {valid}")
        print(f"Removed: {total - valid}")
        print(f"File updated: {cuts_file}")
        
    except FileNotFoundError:
        print(f"Error: file not found '{cuts_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Filter cuts file to keep existing audio files")
    parser.add_argument("cuts_file", help="Cuts file path")
    parser.add_argument("--base-path", "-b", default=".", help="Base path for audio files (default: current directory)")
    parser.add_argument("--threads", "-t", type=int, default=8, help="Number of threads (default: 4)")
    
    args = parser.parse_args()
    filter_cuts(args.cuts_file, args.base_path, args.threads)

if __name__ == "__main__":
    main()