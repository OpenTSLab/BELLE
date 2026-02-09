#!/usr/bin/env python3
"""
Validation script: check prompt information in LibriSpeech data.
Validate whether each item in filter_all_with_prompts.jsonl.gz contains a prompt.
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_prompts(input_file: str) -> Dict[str, Any]:
    """
    Validate prompt info for each item in the file.

    Args:
        input_file: Input file path.

    Returns:
        Statistics dictionary.
    """
    logger.info(f"Validating prompts in {input_file}")
    
    if input_file.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    # Statistics
    stats = {
        'total_items': 0,
        'items_with_prompt': 0,
        'items_without_prompt': 0,
        'items_without_supervision': 0,
        'items_without_custom': 0,
        'prompt_details': {
            'with_source': 0,
            'with_text': 0,
            'with_duration': 0,
            'with_tokens': 0,
        }
    }
    
    items_without_prompt = []
    
    with open_func(input_file, mode, encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Validating"), 1):
            if not line.strip():
                continue
            
            stats['total_items'] += 1
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                continue
            
            # Check supervisions
            supervisions = item.get('supervisions', [])
            if len(supervisions) == 0:
                stats['items_without_supervision'] += 1
                items_without_prompt.append({
                    'line': line_num,
                    'reason': 'no_supervision',
                    'recording_id': item.get('recording', {}).get('id', 'unknown')
                })
                continue
            
            supervision = supervisions[0]
            
            # Check custom field
            custom = supervision.get('custom')
            if custom is None:
                stats['items_without_custom'] += 1
                items_without_prompt.append({
                    'line': line_num,
                    'reason': 'no_custom',
                    'recording_id': item.get('recording', {}).get('id', 'unknown'),
                    'speaker': supervision.get('speaker', 'unknown')
                })
                continue
            
            # Check prompt field
            prompt = custom.get('prompt')
            if prompt is None:
                stats['items_without_prompt'] += 1
                items_without_prompt.append({
                    'line': line_num,
                    'reason': 'no_prompt',
                    'recording_id': item.get('recording', {}).get('id', 'unknown'),
                    'speaker': supervision.get('speaker', 'unknown')
                })
                continue
            
            # Item with prompt
            stats['items_with_prompt'] += 1
            
            # Check prompt details
            if 'source' in prompt and prompt['source']:
                stats['prompt_details']['with_source'] += 1
            if 'text' in prompt and prompt['text']:
                stats['prompt_details']['with_text'] += 1
            if 'duration' in prompt:
                stats['prompt_details']['with_duration'] += 1
            if 'tokens' in prompt and prompt['tokens']:
                stats['prompt_details']['with_tokens'] += 1
    
    # Print statistics
    logger.info("=" * 60)
    logger.info("Validation Results:")
    logger.info("=" * 60)
    logger.info(f"Total items: {stats['total_items']}")
    logger.info(f"Items with prompt: {stats['items_with_prompt']} ({stats['items_with_prompt']/stats['total_items']*100:.2f}%)")
    logger.info(f"Items without supervision: {stats['items_without_supervision']}")
    logger.info(f"Items without custom: {stats['items_without_custom']}")
    logger.info(f"Items without prompt: {stats['items_without_prompt']}")
    logger.info("")
    logger.info("Prompt field details (among items with prompt):")
    logger.info(f"  With source: {stats['prompt_details']['with_source']} ({stats['prompt_details']['with_source']/max(stats['items_with_prompt'], 1)*100:.2f}%)")
    logger.info(f"  With text: {stats['prompt_details']['with_text']} ({stats['prompt_details']['with_text']/max(stats['items_with_prompt'], 1)*100:.2f}%)")
    logger.info(f"  With duration: {stats['prompt_details']['with_duration']} ({stats['prompt_details']['with_duration']/max(stats['items_with_prompt'], 1)*100:.2f}%)")
    logger.info(f"  With tokens: {stats['prompt_details']['with_tokens']} ({stats['prompt_details']['with_tokens']/max(stats['items_with_prompt'], 1)*100:.2f}%)")
    logger.info("=" * 60)
    
    # If there are items without prompt, print the first 10
    if items_without_prompt:
        logger.warning(f"\nFound {len(items_without_prompt)} items without prompt")
        logger.warning("First 10 items without prompt:")
        for i, item_info in enumerate(items_without_prompt[:10], 1):
            logger.warning(f"  {i}. Line {item_info['line']}: {item_info['reason']} - "
                         f"recording_id={item_info['recording_id']}, "
                         f"speaker={item_info.get('speaker', 'N/A')}")
    
    # Determine pass/fail
    if stats['items_with_prompt'] == stats['total_items']:
        logger.info("\n✓ PASS: All items contain prompt!")
    else:
        logger.warning(f"\n✗ FAIL: {stats['total_items'] - stats['items_with_prompt']} items missing prompt")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate prompt information in LibriSpeech data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="egs/librispeech/data/tokenized/vad_lt_14/cuts_train/filter_all_with_prompts.jsonl.gz",
        help="Input jsonl.gz file path"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {args.input}")
        return
    
    logger.info(f"Input file: {args.input}")
    logger.info(f"File size: {input_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Run validation
    stats = validate_prompts(args.input)
    
    logger.info("\nValidation completed!")


if __name__ == "__main__":
    main()
