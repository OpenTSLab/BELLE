from lhotse import CutSet
from tqdm import tqdm
import os
import concurrent.futures
from functools import partial

subset = "train"

input_path = f"data/tokenized/vad_lt_14/cuts_{subset}.jsonl.gz"
tts_models = ["cosyvoice", "f5tts", "indextts", "xtts", "maskgct", "sparktts"]
# tts_models = ["xtts", "maskgct"]

output_path = f"data/tokenized/vad_lt_14/cuts_{subset}/filter_all.jsonl.gz"
# output_path = f"data/tokenized/vad_lt_14/cuts_{subset}/filter_{'_'.join(tts_models)}.jsonl.gz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load the CutSet
cuts = CutSet.from_jsonl(input_path)

def validate_cut(cut, tts_models):
    """Validate that a cut has corresponding audio files for all TTS models."""
    audio_path = cut.recording.sources[0].source
    
    valid = True
    for tts_name in tts_models:
        assert "new_librispeech" in audio_path, f"new_librispeech not found in {audio_path}"
        tts_audio_path = audio_path.replace("new_librispeech", f"librispeech_{tts_name}_vad")

        # Check if the TTS audio file exists
        if not os.path.exists(tts_audio_path):
            # print(f"File not found: {tts_audio_path}")
            valid = False
            break
            
        # Check if the TTS audio is less than 20 seconds
        # try:
        #     from lhotse.audio import Recording
        #     recording = Recording.from_file(tts_audio_path)
        #     if recording.duration > 20.0 or recording.duration < 0.5:
        #         # print(f"Audio too long: {tts_audio_path} ({recording.duration}s)")
        #         valid = False
        #         break
        # except Exception as e:
        #     # print(f"Error checking duration of {tts_audio_path}: {e}")
        #     valid = False
        #     break
    
    return (cut, valid)

# Create a partial function, fixing the tts_models argument
validate_func = partial(validate_cut, tts_models=tts_models)

# Process with multithreading
new_cuts = []
# Set thread count (adjust based on CPU cores)
max_workers = min(32, os.cpu_count() * 2)
print(f"Using {max_workers} threads for processing")

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks and show progress with tqdm
    futures = {executor.submit(validate_func, cut): cut for cut in cuts}
    
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Validating cuts"):
        cut, exists = future.result()
        if exists:
            new_cuts.append(cut)

# Save the new CutSet
new_cuts = CutSet.from_cuts(new_cuts)
print(f"Filtered cuts saved to {output_path}")
# Check the number of cuts in the original and filtered CutSet
print(f"Original cuts: {len(cuts)}")
print(f"Filtered cuts: {len(new_cuts)}")
new_cuts.to_file(output_path)