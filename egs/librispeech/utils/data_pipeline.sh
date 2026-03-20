#!/bin/bash

cd ~/BELLE/egs/librispeech
source ~/miniconda3/bin/activate belle

# Step 0: download LibriSpeech data and unpack it to BELLE/egs/librispeech/download/LibriSpeech

# Step 1: Prepare LibriSpeech to get data/tokenized/cuts_train.jsonl.gz and other splits, text is tokenized by g2p to get phonemes
bash utils/prepare_librispeech.sh

# Step 2: Use silero_vad to remove silence in the audio, and save the processed audio to download/new_librispeech
python utils/vad.py download/LibriSpeech/ download/new_librispeech/

# Step 3: Get the .jsonl.gz file for new_librispeech
python utils/update_cuts_for_vad.py \
    --input data/tokenized/cuts_train.jsonl.gz \
    --output data/tokenized/cuts_train_vad.jsonl.gz \
    --old-root download/LibriSpeech \
    --new-root download/new_librispeech \
    --drop-missing

# Step 4: Filter by duration to get data/tokenized/vad_lt_14/cuts_train.jsonl.gz
python utils/filter_duration.py

#--------------------------------------------------------------------
# The following steps 5-7 are for data augmentation using TTS, which is optional but can help to improve the robustness of BELLE. You can skip these steps and directly use the original audio for training BELLE.
#--------------------------------------------------------------------

# Step 5: Synthesize audio for the subset of cuts_train, using different TTS models, and save them to download/librispeech_{tts_name}/
tts=("cosyvoice" "f5tts" "indextts" "xtts" "maskgct" "sparktts")
for t in ${tts[@]}; do
    bash ../../tts-launch/bash_scripts/${t}_infer.sh
done

# Step 6: Use silero_vad to remove silence in the audio, and save the processed audio to download/librispeech_{tts_name}_vad/
bash utils/vad_tts.sh

# Step 7: Filter the data/tokenized/vad_lt_14/cuts_train.jsonl.gz to contain only successful items (some items may fail during TTS), and save to data/tokenized/vad_lt_14/cuts_train/filter_all.jsonl.gz
python utils/filter_data.py

#---------------------------------------------------------------------

# Step 8: Now you can use data/tokenized/vad_lt_14/cuts_train/filter_all.jsonl.gz for training your TTS model.
# If synthesized data of TTS is not used, you can directly use data/tokenized/vad_lt_14/cuts_train.jsonl.gz for training your TTS model.

#---------------------------------------------------------------------
# The following are the content of the BELLE-stream
#---------------------------------------------------------------------

# Step 9: Prepare .jsonl.gz with audio prompts for BELLE-stream training
python utils/process_librispeech_with_prompts.py \
    --input data/tokenized/vad_lt_14/cuts_train/filter_all.jsonl.gz

# Step 10: Use data/tokenized/vad_lt_14/cuts_train/filter_all_with_prompts.jsonl.gz for BELLE-stream training. Use the last checkpoint of BELLE to initialize BELLE-stream. Specifically, you should copy the last checkpoint of BELLE to the directory of BELLE-stream, and rename it to epoch-1.pt, which will be automatically loaded when training BELLE-stream (set start_epoch=2 and train_stage=2, refer to scripts/run_stream.sh for details).