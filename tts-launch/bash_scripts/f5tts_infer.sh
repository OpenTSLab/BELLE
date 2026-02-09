MODEL_NAME="f5tts"
PYTHON_BIN="~/miniconda3/envs/f5-tts/bin/python3.10"

cd ~/BELLE

$PYTHON_BIN tts-launch/${MODEL_NAME}_launch.py \
    --recordings egs/librispeech/data/tokenized/vad_lt_14/cuts_train.jsonl.gz \
    --output_dir egs/librispeech/download/librispeech_f5tts/ \
    --audio_base_dir egs/librispeech \
    --processes_per_gpu 4