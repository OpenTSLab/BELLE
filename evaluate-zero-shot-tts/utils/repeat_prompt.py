#!/usr/bin/env python3
"""
Script purpose:
Read files from the source folder. If a filename ends with _1 or _2, replace
its content with the corresponding _0 file content, while keeping the original
filename, and copy it to the target folder.
"""

import os
import shutil
from pathlib import Path

def process_files(source_dir, target_dir):
    """
    Process files:
    - If a file ends with _1 or _2, replace it with the corresponding _0 content
    - Copy to the target folder while keeping the original filename
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Ensure the target folder exists
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    processed_count = 0
    total_count = 0
    
    # Iterate over all files in the source folder
    for file_path in source_path.iterdir():
        if file_path.is_file():
            total_count += 1
            file_name = file_path.name
            
            # Check if the filename ends with _1 or _2 (before the extension)
            stem = file_path.stem  # Filename without extension
            suffix = file_path.suffix  # File extension
            
            if stem.endswith('_1') or stem.endswith('_2'):
                # Build corresponding _0 filename
                base_name = stem[:-2] + '_0'  # Remove _1 or _2, then add _0
                zero_file_name = base_name + suffix
                zero_file_path = source_path / zero_file_name
                
                if zero_file_path.exists():
                    # Copy _0 content to target folder but keep original filename
                    target_file_path = target_path / file_name
                    shutil.copy2(zero_file_path, target_file_path)
                    print(f"Processed file: {file_name} -> using content from {zero_file_name}")
                    processed_count += 1
                else:
                    print(f"Warning: corresponding _0 file not found: {zero_file_name}")
                    # If _0 file is missing, copy the original file
                    target_file_path = target_path / file_name
                    shutil.copy2(file_path, target_file_path)
            else:
                # For other files, copy directly
                target_file_path = target_path / file_name
                shutil.copy2(file_path, target_file_path)
    
    print("\nProcessing complete!")
    print(f"Total files: {total_count}")
    print(f"Processed files: {processed_count}")
    print(f"Target folder: {target_dir}")

def main():
    # Source folder path
    source_dir = "evaluate-zero-shot-tts/evalsets/librispeech-test-clean/exp_aligned_pl3_r3"
    
    # Target folder path
    target_dir = "evaluate-zero-shot-tts/evalsets/librispeech-test-clean/exp_aligned_pl3_r3_processed"
    
    # Check if the source folder exists
    if not os.path.exists(source_dir):
        print(f"Error: source folder does not exist: {source_dir}")
        return
    
    print(f"Source folder: {source_dir}")
    print(f"Target folder: {target_dir}")
    print("Starting to process files...")
    
    # Process files
    process_files(source_dir, target_dir)

if __name__ == "__main__":
    main()