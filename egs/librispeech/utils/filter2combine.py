import os
import concurrent.futures
from tqdm import tqdm
import soundfile as sf
from lhotse import CutSet
from functools import partial
import copy

# Changed from single model to a list of models
# tts_models = ["cosyvoice", "f5tts", "indextts", "xtts", "maskgct", "sparktts"]
tts_models = ["xtts", "maskgct"]
# subset = "train_clean_100"
# subset = "train_clean_360"
subset = "train_clean_100"

def get_audio_duration(audio_path: str) -> float:
    """Read an audio file and return the actual duration."""
    try:
        info = sf.info(audio_path)
        duration = info.duration
        num_samples = info.frames if info.frames is not None else int(duration * info.samplerate)
        return duration, num_samples
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None, None

def update_cut_for_all_models(cut, base_dir: str, tts_models: list):
    """
    Update a single cut for all TTS models and return a list containing the
    original cut and all TTS model cuts.
    """
    result_cuts = []
    
    # Add the original cut first (do not modify the path)
    original_cut = copy.deepcopy(cut)
    result_cuts.append(original_cut)
    
    # Create a corresponding cut for each TTS model
    for tts_model in tts_models:
        try:
            # Deep copy the original cut
            tts_cut = copy.deepcopy(cut)
            
            # Replace path
            old_source = tts_cut.recording.sources[0].source
            new_source = old_source.replace("new_librispeech", f"librispeech_{tts_model}_vad")
            tts_cut.recording.sources[0].source = new_source
            
            # Get full audio file path
            audio_path = os.path.join(base_dir, new_source)
            
            # Read audio file to get actual duration
            actual_duration, actual_num_samples = get_audio_duration(audio_path)
            if actual_duration is None:
                print(f"Skipping {tts_model} model for {cut.id}: failed to read audio file")
                continue
                
            # Update duration
            tts_cut.duration = actual_duration
            tts_cut.recording.duration = actual_duration
            tts_cut.recording.num_samples = actual_num_samples
            
            # Update supervision duration
            for supervision in tts_cut.supervisions:
                supervision.duration = actual_duration
            
            result_cuts.append(tts_cut)
            
        except Exception as e:
            print(f"Error processing cut {cut.id} for model {tts_model}: {e}")
            continue
    
    return result_cuts

def process_with_progress(cut_list, update_func, num_workers):
    """Process cut_list with multithreading and show a progress bar."""
    total = len(cut_list)
    
    # Create progress bar updater
    pbar = tqdm(total=total, desc="Processing")
    
    def process_and_update(cut):
        result = update_func(cut)
        pbar.update(1)
        return result
    
    # Process in parallel with ThreadPoolExecutor.map
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_and_update, cut_list))
    
    pbar.close()
    
    # Flatten results (each cut returns a list)
    flattened_results = []
    for result_list in results:
        if result_list is not None:
            flattened_results.extend(result_list)
    
    print(f"Processed {total} original cuts, generated {len(flattened_results)} total cuts")
    return flattened_results

def process_jsonl_file(input_file: str, base_dir: str, tts_models: list, num_workers: int = 8):
    """Process JSONL cuts with multithreading and generate model variants for each cut."""
    # Load CutSet
    cuts = CutSet.from_jsonl(input_file)
    
    total_cuts = len(cuts)
    print(f"Loaded {total_cuts} cuts in total")
    
    # Prepare update function
    update_func = partial(update_cut_for_all_models, base_dir=base_dir, tts_models=tts_models)
    
    # Convert to list for processing
    cut_list = list(cuts)
    
    # Process with multithreaded map
    all_cuts_list = process_with_progress(cut_list, update_func, num_workers)
    
    print(f"Completed! Generated {len(all_cuts_list)} cuts in total")
    
    return all_cuts_list

if __name__ == "__main__":
    input_file = f"data/tokenized/vad_lt_14/cuts_{subset}/filter_xtts_maskgct.jsonl.gz"
    
    print("Starting to process all models...")
    
    # Process all models at once; results are naturally interleaved
    all_cuts = process_jsonl_file(
        input_file=input_file,
        base_dir="",
        tts_models=tts_models,
        num_workers=os.cpu_count()
    )
    
    # Create combined CutSet
    combined_cutset = CutSet.from_cuts(all_cuts)
    print(f"Completed! Total cuts: {len(combined_cutset)}")
    
    # Write final output
    output_file = f"data/tokenized/vad_lt_14/cuts_{subset}/combined_xtts_maskgct.jsonl.gz"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_cutset.to_file(output_file)
    
    print(f"Completed! Results written to: {output_file}")