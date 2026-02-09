#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to pick the best audio sample for each speaker based on MOS scores
"""

import os
import shutil
import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_mos_scores(file_path):
    """
    Parse MOS scores from file
    
    Args:
        file_path: Path to the file containing MOS scores
        
    Returns:
        dict: Dictionary with wav_name as key and score as value
    """
    scores = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Skip first line (average score)
    for line in lines[1:]:
        line = line.strip()
        if line and ':' in line:
            wav_name, score_str = line.split(':', 1)
            wav_name = wav_name.strip()
            score = float(score_str.strip())
            scores[wav_name] = score
            
    return scores


def parse_wer_file(file_path):
    """
    Parse WER file to find audio files with WER=0
    
    Args:
        file_path: Path to the WER file
        
    Returns:
        set: Set of wav file names with WER=0
    """
    zero_wer_files = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the second occurrence of "================"
    separator = "=" * 80
    parts = content.split(separator)
    
    if len(parts) < 3:
        print("Warning: Could not find the second separator in WER file")
        return zero_wer_files
    
    # Process the part after the second separator
    alignments_section = separator.join(parts[2:])
    
    # Find all utterance entries with WER 0.00
    pattern = r'(\S+_wav_\w+_\d+),\s*%WER\s+0\.00'
    matches = re.findall(pattern, alignments_section)
    
    for match in matches:
        wav_name = match + '.wav'
        zero_wer_files.add(wav_name)
        # print(f"Found WER=0 file: {wav_name}")
    
    return zero_wer_files


def group_by_speaker(scores, zero_wer_files=None):
    """
    Group audio files by speaker ID (first part before '-')
    Filter by zero WER files if provided
    
    Args:
        scores: Dictionary with wav_name as key and score as value
        zero_wer_files: Set of wav files with WER=0 (optional)
        
    Returns:
        dict: Dictionary with speaker_id as key and list of (wav_name, score) as value
    """
    speaker_groups = defaultdict(list)
    
    for wav_name, score in scores.items():
        # If zero_wer_files is provided, only include files with WER=0
        if zero_wer_files is not None and wav_name not in zero_wer_files:
            continue
            
        # Extract speaker ID from wav name (first part before '-')
        speaker_id = wav_name.split('-')[0]
        speaker_groups[speaker_id].append((wav_name, score))
        
    return speaker_groups


def select_best_audio_per_speaker(speaker_groups):
    """
    Select the audio with highest MOS score for each speaker
    
    Args:
        speaker_groups: Dictionary with speaker_id as key and list of (wav_name, score) as value
        
    Returns:
        dict: Dictionary with speaker_id as key and (best_wav_name, best_score) as value
    """
    best_audios = {}
    
    for speaker_id, audios in speaker_groups.items():
        # Find the audio with highest score
        best_wav_name, best_score = max(audios, key=lambda x: x[1])
        best_audios[speaker_id] = (best_wav_name, best_score)
        
    return best_audios


def copy_selected_audios(best_audios, source_dir, target_dir):
    """
    Copy selected audio files to target directory
    
    Args:
        best_audios: Dictionary with speaker_id as key and (wav_name, score) as value
        source_dir: Source directory containing original audio files
        target_dir: Target directory to copy selected files
        
    Returns:
        list: List of successfully copied wav file names
    """
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    for speaker_id, (wav_name, score) in best_audios.items():
        # Copy wav file
        wav_source_path = os.path.join(source_dir, wav_name)
        wav_target_path = os.path.join(target_dir, wav_name)
        
        # Copy corresponding txt file
        txt_name = wav_name.replace('.wav', '.txt')
        txt_source_path = os.path.join(source_dir, txt_name)
        txt_target_path = os.path.join(target_dir, txt_name)
        
        wav_copied = False
        txt_copied = False
        
        # Copy wav file
        if os.path.exists(wav_source_path):
            try:
                shutil.copy2(wav_source_path, wav_target_path)
                copied_files.append(wav_name)
                wav_copied = True
                print(f"Copied {wav_name} (Speaker: {speaker_id}, Score: {score:.4f})")
            except Exception as e:
                print(f"Error copying {wav_name}: {e}")
        else:
            print(f"Warning: Source wav file not found: {wav_source_path}")
        
        # Copy txt file
        if os.path.exists(txt_source_path):
            try:
                shutil.copy2(txt_source_path, txt_target_path)
                txt_copied = True
                print(f"Copied {txt_name} (corresponding text file)")
            except Exception as e:
                print(f"Error copying {txt_name}: {e}")
        else:
            print(f"Warning: Source txt file not found: {txt_source_path}")
        
        if not wav_copied and not txt_copied:
            print(f"Warning: Neither wav nor txt file found for {wav_name}")
            
    return copied_files


def save_wav_list(wav_names, output_file):
    """
    Save list of wav file names to a text file
    
    Args:
        wav_names: List of wav file names
        output_file: Path to output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for wav_name in sorted(wav_names):
            f.write(f"{wav_name}\n")
    
    print(f"Saved wav list to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Pick best audio sample for each speaker based on MOS scores"
    )
    parser.add_argument(
        "--scores_file", 
        required=True,
        help="Path to file containing MOS scores"
    )
    parser.add_argument(
        "--wer_file",
        help="Path to WER file (optional, to filter by WER=0)"
    )
    parser.add_argument(
        "--source_dir", 
        required=True,
        help="Source directory containing audio files"
    )
    parser.add_argument(
        "--target_dir", 
        required=True,
        help="Target directory to copy selected files"
    )
    parser.add_argument(
        "--output_list", 
        default="selected_wav_list.txt",
        help="Output file for wav names list (default: selected_wav_list.txt)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Picking best audio samples for each speaker")
    print("=" * 60)
    
    # Parse MOS scores
    print(f"Parsing MOS scores from: {args.scores_file}")
    scores = parse_mos_scores(args.scores_file)
    print(f"Found {len(scores)} audio files with scores")
    
    # Parse WER file if provided
    zero_wer_files = None
    if args.wer_file:
        print(f"\nParsing WER file: {args.wer_file}")
        zero_wer_files = parse_wer_file(args.wer_file)
        print(f"Found {len(zero_wer_files)} files with WER=0")
        
        # Filter scores to only include WER=0 files
        filtered_scores = {k: v for k, v in scores.items() if k in zero_wer_files}
        print(f"After WER=0 filtering: {len(filtered_scores)} files remain")
        scores = filtered_scores
    
    # Group by speaker
    speaker_groups = group_by_speaker(scores, zero_wer_files)
    print(f"Found {len(speaker_groups)} unique speakers")
    
    # Select best audio per speaker
    best_audios = select_best_audio_per_speaker(speaker_groups)
    
    print("\nSelected best audio for each speaker:")
    for speaker_id, (wav_name, score) in best_audios.items():
        wer_info = " (WER=0)" if zero_wer_files and wav_name in zero_wer_files else ""
        print(f"  Speaker {speaker_id}: {wav_name} (Score: {score:.4f}){wer_info}")
    
    # Copy selected audios
    print(f"\nCopying selected files from {args.source_dir} to {args.target_dir}...")
    copied_files = copy_selected_audios(best_audios, args.source_dir, args.target_dir)
    
    # Save wav list
    save_wav_list(copied_files, args.output_list)
    
    print(f"\nCompleted! Copied {len(copied_files)} files.")
    print("=" * 60)


if __name__ == "__main__":
    main()