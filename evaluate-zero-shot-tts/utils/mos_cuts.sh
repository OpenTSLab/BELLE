cd ~/BELLE/evaluate-zero-shot-tts

source ~/miniconda3/bin/activate belle

torchrun utils/mos_cuts.py \
  --jsonl_path ../egs/librispeech/data/tokenized/speaker_39.jsonl.gz \
  --out_path ../egs/librispeech/data/tokenized/speaker_39_mos.jsonl.gz \
  --bs 1 \
  --num_workers 8