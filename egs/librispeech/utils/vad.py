import os
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str, help="Input directory containing original FLAC files")
parser.add_argument("output_dir", type=str, help="Output directory to save processed audio files")

args = parser.parse_args()

# Set input/output paths
input_dir = args.input_dir
print(f"Input directory: {input_dir}")
output_dir = args.output_dir

# Load Silero VAD model
model = load_silero_vad()

# Silence threshold and other parameters
speech_pad = 0  # Keep this much silence around speech segments (seconds)

def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

def find_flac_files(input_dir):
    """Find all FLAC files."""
    flac_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac'):
                flac_files.append(os.path.join(root, file))
    return flac_files

def process_audio_file(audio_path):
    """Process a single audio file, detect speech regions, and concatenate without silences."""
    try:
        # Read audio
        wav = read_audio(audio_path)
        sr = sf.info(audio_path).samplerate
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav, 
            model,
            return_seconds=False  # Return sample indices instead of seconds
        )
        
        if not speech_timestamps:
            print(f"No speech detected: {audio_path}")
            return None  # No speech detected
        
        # Prepare to concatenate all speech segments
        concat_segments = []
        total_duration = 0
        
        for ts in speech_timestamps:
            # Add surrounding silence (in samples)
            pad_samples = int(speech_pad * sr)
            start_idx = max(0, ts['start'] - pad_samples)
            end_idx = min(len(wav), ts['end'] + pad_samples)
            
            # Slice audio segment
            segment = wav[start_idx:end_idx]
            concat_segments.append(segment)
            total_duration += (end_idx - start_idx) / sr
        
        # Concatenate all speech segments
        concatenated_audio = np.concatenate(concat_segments)
        
        # Build new filename and path
        rel_path = os.path.relpath(audio_path, input_dir)
        new_audio_path = os.path.join(output_dir, rel_path)
        ensure_dir(os.path.dirname(new_audio_path))

        # Save new audio
        sf.write(new_audio_path, concatenated_audio, sr)
        
        return new_audio_path, total_duration
    
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None

print(f"Searching for audio files in {input_dir}...")
flac_files = find_flac_files(input_dir)
print(f"Found {len(flac_files)} FLAC files")

if len(flac_files) == 0:
    print(f"Warning: no FLAC files found in {input_dir}")
    exit(1)

print("Starting audio segmentation...")
# Process in parallel with ProcessPoolExecutor
processed_files = []
total_input_duration = 0
total_output_duration = 0

with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_audio_file, flac_files), total=len(flac_files)))
    
    for result in results:
        if result:
            file_path, duration = result
            processed_files.append(file_path)
            total_output_duration += duration
            
            # Compute original file duration
            original_info = sf.info(flac_files[len(processed_files)-1])
            total_input_duration += original_info.duration

# Print statistics
print(f"Completed: processed {len(processed_files)}/{len(flac_files)} files")
print(f"Total original audio duration: {total_input_duration:.2f} s")
print(f"Total processed audio duration: {total_output_duration:.2f} s")
print(f"Duration change: {(total_input_duration - total_output_duration) / total_input_duration * 100:.2f}%")
print(f"Processed files saved to: {output_dir}")