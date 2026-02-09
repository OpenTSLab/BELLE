#!/usr/bin/env python3
"""
Data processing script: add prompt information for LibriSpeech data.
Process filter_all.jsonl.gz, filter audio with duration in 3–8 seconds to
build a speaker dictionary, then randomly select one prompt per item.
"""

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl_gz(input_file: str) -> List[Dict[str, Any]]:
    """Load a jsonl.gz file."""
    logger.info(f"Loading data from {input_file}")
    data = []
    
    if input_file.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(input_file, mode, encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            if line.strip():
                data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} items")
    return data


def build_speaker_dict(data: List[Dict[str, Any]], min_duration: float = 3.0, max_duration: float = 8.0) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a speaker dictionary containing only audio within the duration range.

    Args:
        data: Data list.
        min_duration: Minimum duration (seconds).
        max_duration: Maximum duration (seconds).

    Returns:
        speaker_dict: {speaker_id: [prompt_info_list]}
    """
    logger.info(f"Building speaker dictionary (duration: {min_duration}s - {max_duration}s)")
    speaker_dict = defaultdict(list)
    
    count = 0
    filtered_count = 0
    
    for item in tqdm(data, desc="Building speaker dict"):
        count += 1
        
        # Check if duration is within range
        duration = item.get('duration', 0)
        if not (min_duration <= duration <= max_duration):
            continue
        
        filtered_count += 1
        
        # Get speaker info
        supervisions = item.get('supervisions', [])
        if len(supervisions) == 0:
            continue
        
        supervision = supervisions[0]
        speaker_id = supervision.get('speaker')
        
        if speaker_id is None:
            continue
        
        # Get audio file path
        recording = item.get('recording', {})
        sources = recording.get('sources', [])
        audio_source = None
        if sources and len(sources) > 0:
            # Get the first source path
            audio_source = sources[0].get('source', '')
        
        # Extract prompt info
        prompt_info = {
            "source": audio_source,
            "text": supervision.get('text', ''),
            "duration": duration,
        }
        
        # Add tokens info if available
        custom = supervision.get('custom', {})
        if custom and 'tokens' in custom:
            prompt_info["tokens"] = custom['tokens']['text']
        
        speaker_dict[speaker_id].append(prompt_info)
    
    logger.info(f"Processed {count} items, filtered {filtered_count} items within duration range")
    logger.info(f"Built speaker dictionary with {len(speaker_dict)} speakers")
    
    # Print some statistics
    prompt_counts = [len(prompts) for prompts in speaker_dict.values()]
    if prompt_counts:
        logger.info(f"Average prompts per speaker: {sum(prompt_counts) / len(prompt_counts):.2f}")
        logger.info(f"Min prompts per speaker: {min(prompt_counts)}")
        logger.info(f"Max prompts per speaker: {max(prompt_counts)}")
    
    return dict(speaker_dict)


def add_prompts_to_data(
    data: List[Dict[str, Any]],
    speaker_dict: Dict[str, List[Dict[str, Any]]],
    output_file: str,
    seed: int = 42
):
    """
    Add prompt information for each item.

    Args:
        data: Data list.
        speaker_dict: Speaker dictionary.
        output_file: Output file path.
        seed: Random seed.
    """
    logger.info(f"Adding prompts to data and saving to {output_file}")
    random.seed(seed)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine whether gzip compression is needed
    if output_file.endswith('.gz'):
        open_func = gzip.open
        mode = 'wt'
    else:
        open_func = open
        mode = 'w'
    
    with open_func(output_file, mode, encoding='utf-8') as out_f:
        count = 0
        added_prompt_count = 0
        no_speaker_count = 0
        no_prompt_available_count = 0
        
        for item in tqdm(data, desc="Adding prompts"):
            count += 1
            
            # Get speaker info
            supervisions = item.get('supervisions', [])
            if len(supervisions) == 0:
                no_speaker_count += 1
                # No supervision; write the original item
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                continue
            
            supervision = supervisions[0]
            speaker_id = supervision.get('speaker')
            
            if speaker_id is None or speaker_id not in speaker_dict:
                no_speaker_count += 1
                # No speaker or speaker not in dictionary; write the original item
                # out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                continue
            
            # Randomly choose a prompt from the speaker dictionary
            available_prompts = speaker_dict[speaker_id]
            
            if len(available_prompts) == 0:
                no_prompt_available_count += 1
                # No available prompt; write the original item
                # out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                continue
            
            # Randomly select a prompt
            selected_prompt = random.choice(available_prompts)
            
            # Add prompt to supervision custom field
            if 'custom' not in supervision:
                supervision['custom'] = {}
            
            supervision['custom']['prompt'] = selected_prompt
            
            added_prompt_count += 1
            
            # Write the updated item
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Processed {count} items")
        logger.info(f"Added prompts to {added_prompt_count} items")
        logger.info(f"Skipped {no_speaker_count} items without speaker")
        logger.info(f"Skipped {no_prompt_available_count} items without available prompts")
    
    logger.info(f"Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add prompt information for LibriSpeech data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/tokenized/cuts_train_split_mos_filter_4.3.jsonl.gz",
        help="Input jsonl.gz file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output jsonl.gz file path (default: create *_with_prompts.jsonl.gz in the input directory)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum duration (seconds) when building speaker dictionary"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=8.0,
        help="Maximum duration (seconds) when building speaker dictionary"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        input_path = Path(args.input)
        if args.input.endswith('.jsonl.gz'):
            output_filename = input_path.name.replace('.jsonl.gz', '_with_prompts.jsonl.gz')
        elif args.input.endswith('.jsonl'):
            output_filename = input_path.name.replace('.jsonl', '_with_prompts.jsonl')
        else:
            output_filename = input_path.name + '_with_prompts.jsonl.gz'
        args.output = str(input_path.parent / output_filename)
    
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Duration range: {args.min_duration}s - {args.max_duration}s")
    logger.info(f"Random seed: {args.seed}")
    
    # Load data
    data = load_jsonl_gz(args.input)
    
    # Build speaker dictionary
    speaker_dict = build_speaker_dict(
        data, 
        min_duration=args.min_duration, 
        max_duration=args.max_duration
    )
    
    # Add prompts and save
    add_prompts_to_data(
        data, 
        speaker_dict, 
        args.output,
        seed=args.seed
    )
    
    logger.info("Processing completed!")


if __name__ == "__main__":
    main()
