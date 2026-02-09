MODEL_NAME="spark_tts"
PYTHON_BIN="~/miniconda3/envs/sparktts/bin/python3.12"

cd ~/BELLE

$PYTHON_BIN tts-launch/${MODEL_NAME}_launch.py \
    --recordings egs/librispeech/data/tokenized/vad_lt_14/cuts_train.jsonl.gz \
    --output_dir egs/librispeech/download/librispeech_sparktts/ \
    --audio_base_dir egs/librispeech \
    --processes_per_gpu 4