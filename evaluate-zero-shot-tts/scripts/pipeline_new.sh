cd ~/BELLE/evaluate-zero-shot-tts
source ~/miniconda3/bin/activate belle
export PYTHONPATH=~/BELLE/icefall:$PYTHONPATH

exps="belle-lr5e-4-kl0-edl0.2-flux0.5-epoch60-cuts_train_filter_all-tts_models_cosyvoice_indextts_sparktts_f5tts_xtts_maskgct-loss_weight_0.22_0.13_0.13_0.13_0.13_0.13_0.13-coef0.5-inversegamma
"

dataset=librispeech
backend=espeak
samples_dir=samples_new
mkdir -p $samples_dir
results_dir=results_new
mkdir -p $results_dir

for task in cont; do
    for epoch in 60; do
        for exp in $exps; do
            torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK  --nproc_per_node=$PROC_PER_NODE --master_addr=$MASTER_ADDR --master_port=29500 inference_new.py --model_name melle --task_key $task --ckpt ../egs/$dataset/exp/$exp/epoch-$epoch.pt --backend $backend --output_dir $samples_dir --dataset_name $dataset --results_dir $results_dir

            # If $task is "cross", then replace corresponding_target
            if [ "$task" = "cross" ]; then
                corresponding_target="wav_pg"
                eval_data="exp_aligned_pl3_r3"
            else
                corresponding_target="wav_c"
                eval_data="exp_base_pl3_r3"
            fi
            
            python evaluate_new.py -m wer -e hubert -t $corresponding_target -d $samples_dir/librispeech-test-clean/$eval_data/melle/$task/$exp/"epoch-"$epoch -o $results_dir

            python evaluate_new.py -m wer -e conformer -t $corresponding_target -d $samples_dir/librispeech-test-clean/$eval_data/melle/$task/$exp/"epoch-"$epoch -o $results_dir

            python evaluate_new.py -m sim_o -e wavlmuni -t $corresponding_target -d $samples_dir/librispeech-test-clean/$eval_data/melle/$task/$exp/"epoch-"$epoch -o $results_dir

            python evaluate_new.py -m sim_r -e wavlmuni -t $corresponding_target -d $samples_dir/librispeech-test-clean/$eval_data/melle/$task/$exp/"epoch-"$epoch -o $results_dir
            
            python utmos/predict_speechmos.py --mode predict_dir --inp_dir $samples_dir/librispeech-test-clean/$eval_data/melle/$task/$exp/"epoch-"$epoch --bs 1 --out_path $results_dir/librispeech-test-clean/melle/$eval_data/$task/$exp/"epoch-"$epoch/mos.txt --task $corresponding_target
        done
    done
done
