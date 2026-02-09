source ~/BELLE/scripts/settings_stream.sh

cd ~/BELLE/egs/librispeech
source ~/miniconda3/bin/activate belle
export PYTHONPATH=~/BELLE/icefall:$PYTHONPATH
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK  --nproc_per_node=$PROC_PER_NODE --master_addr=$MASTER_ADDR --master_port=29500 bin/trainer.py \
    --max-duration $max_duration --train-stage $train_stage \
    --num-buckets 30 --dtype "float32" --save-every-n 20000 --num-workers 16 \
    --model-name $model_name --norm-first true --pos-learn $pos_learn \
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
    --base-lr $lr --warmup-ratio $warmup_ratio --average-period 0 --optimizer-name $optimizer_name \
    --num-epochs $num_epochs --start-epoch $start_epoch --start-batch 0 --accumulate-grad-steps 1 \
    --exp-dir exp/$exp_name --oom-check False --exp-name $exp_name --kl-loss-weight $kl_loss_weight --flux-loss-weight $flux_loss_weight --clip $clip --scheduler $scheduler --edl-loss-weight $edl_loss_weight --steps-per-epoch $steps_per_epoch --sampling-rate 16000 --tensorboard True --perturb-speed $perturb_speed --train-cuts-path $train_cuts_path  --tts-models "${tts_models[@]}" --loss-weight "${loss_weight[@]}" --coef $coef --sample-method $sample_method --num-text-tokens $num_text_tokens --audio-embeddings $audio_embeddings --text-embeddings $text_embeddings --text-tokens $text_tokens --dataset $dataset --n-frames $n_frames --stream-mode $stream --text-chunk-size $text_chunk_size --audio-chunk-size $audio_chunk_size
