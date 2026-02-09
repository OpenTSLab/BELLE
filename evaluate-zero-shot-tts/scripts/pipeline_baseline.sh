cd ~/BELLE/evaluate-zero-shot-tts/
export PYTHONPATH=~/BELLE/icefall:$PYTHONPATH

samples_dir=samples_new2
mkdir -p $samples_dir
results_dir=results_new2
mkdir -p $results_dir
model_name=maskgct
exp=maskgct/maskgct

for task in cross; do
    source ~/miniconda3/bin/activate maskgct
    # torchrun inference_new.py --model_name $model_name --task_key $task --output_dir $samples_dir --ckpt $exp

    # If $task is "cross", then replace corresponding_target
    if [ "$task" = "cross" ]; then
        corresponding_target="wav_pg"
        eval_data="exp_aligned_pl3_r3_processed"
    else
        corresponding_target="wav_c"
        eval_data="exp_base_pl3_r3"
    fi

    source ~/miniconda3/bin/activate belle
    
    # python evaluate_new.py -m wer -e hubert -t $corresponding_target -d $samples_dir/librispeech-test-clean/$eval_data/$model_name/$task/$exp -o $results_dir
    python evaluate_new.py -m wer -e conformer -t $corresponding_target -d $samples_dir/librispeech-test-clean/$eval_data/$model_name/$task/$exp -o $results_dir
    python evaluate_new.py -m sim_o -e wavlmuni -t $corresponding_target -d $samples_dir/librispeech-test-clean/$eval_data/$model_name/$task/$exp -o $results_dir
    python utmos/predict_speechmos.py --mode predict_dir --inp_dir $samples_dir/librispeech-test-clean/$eval_data/$model_name/$task/$exp --bs 1 --out_path $results_dir/librispeech-test-clean/$model_name/$eval_data/$task/$exp/mos.txt --task $corresponding_target
done