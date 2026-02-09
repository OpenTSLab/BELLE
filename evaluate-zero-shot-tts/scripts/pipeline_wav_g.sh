cd ~/BELLE/evaluate-zero-shot-tts/
source ~/miniconda3/bin/activate belle
export PYTHONPATH=~/BELLE/icefall:$PYTHONPATH

# python evaluate.py -m wer -e hubert -t wav_g -d samples/librispeech-test-clean/exp_base_pl3_r3/melle/cont/belle-lr5e-4-kl0-edl0.2-flux0.75-epoch150-vad_lt_14_cuts_train_clean_100_filter_xtts_maskgct-tts_models_xtts_maskgct-loss_weight_0.5_0.25_0.25-coef0.5-inversegamma/epoch-150

# python evaluate.py -m wer -e conformer -t wav_g -d samples/librispeech-test-clean/exp_base_pl3_r3/melle/cont/belle-lr5e-4-kl0-edl0.2-flux0.75-epoch150-vad_lt_14_cuts_train_clean_100_filter_xtts_maskgct-tts_models_xtts_maskgct-loss_weight_0.5_0.25_0.25-coef0.5-inversegamma/epoch-150


# python evaluate.py -m sim_o -e wavlmuni -t wav_g -d samples/librispeech-test-clean/exp_base_pl3_r3/melle/cont/belle-lr5e-4-kl0-edl0.2-flux0.75-epoch150-vad_lt_14_cuts_train_clean_100_filter_xtts_maskgct-tts_models_xtts_maskgct-loss_weight_0.5_0.25_0.25-coef0.5-inversegamma/epoch-150

python evaluate_new.py -m sim_o -e wavlmuni -t wav_g -d samples_new/librispeech-test-clean/exp_aligned_pl3_r3/melle/cross/belle-lr5e-4-kl0-edl0.2-flux0.5-epoch60-cuts_train_filter_all-tts_models_cosyvoice_indextts_sparktts_f5tts_xtts_maskgct-loss_weight_0.22_0.13_0.13_0.13_0.13_0.13_0.13-coef0.5-inversegamma/epoch-60 -o results_new