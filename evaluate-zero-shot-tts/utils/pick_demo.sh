#!/bin/bash
cd evaluate-zero-shot-tts

python pick_demo.py \
    --scores_file results_new/librispeech-test-clean/melle/exp_aligned_pl3_r3/cross/belle-lr1e-4-flux0.5-epoch10-tokenized_cuts_train_split_mos_filter_4.3-tts_models_-loss_weight_1-train_stage2-nframes1/epoch-10/mos.txt \
    --wer_file results_new/librispeech-test-clean/melle/exp_aligned_pl3_r3/cross/belle-lr1e-4-flux0.5-epoch10-tokenized_cuts_train_split_mos_filter_4.3-tts_models_-loss_weight_1-train_stage2-nframes1/epoch-10/conformer_wav_pg_wer.txt \
    --source_dir samples_new/librispeech-test-clean/exp_aligned_pl3_r3/melle/cross/belle-lr1e-4-flux0.5-epoch10-tokenized_cuts_train_split_mos_filter_4.3-tts_models_-loss_weight_1-train_stage2-nframes1/epoch-10 \
    --target_dir demo2/librispeech-test-clean/exp_aligned_pl3_r3/melle/cross/belle-lr1e-4-flux0.5-epoch10-tokenized_cuts_train_split_mos_filter_4.3-tts_models_-loss_weight_1-train_stage2-nframes1/epoch-10 \
    --output_list demo2/librispeech-test-clean/exp_aligned_pl3_r3/melle/cross/belle-lr1e-4-flux0.5-epoch10-tokenized_cuts_train_split_mos_filter_4.3-tts_models_-loss_weight_1-train_stage2-nframes1/epoch-10/list/best_samples.txt


# Example usage of pick_demo.py

# Example 1: Basic usage (without WER filtering)
# python pick_demo.py \
#     --scores_file "results/librispeech-test-clean/melle/scores.txt" \
#     --source_dir "samples/librispeech-test-clean/exp_aligned_pl3_r3/" \
#     --target_dir "selected_samples/" \
#     --output_list "best_samples.txt"

# Example 2: With WER filtering (only select from WER=0 files)
# python pick_demo.py \
#     --scores_file "results/librispeech-test-clean/melle/scores.txt" \
#     --wer_file "results/librispeech-test-clean/melle/wer_results.txt" \
#     --source_dir "samples/librispeech-test-clean/exp_aligned_pl3_r3/" \
#     --target_dir "selected_samples_wer0/" \
#     --output_list "best_samples_wer0.txt"

# Example 3: Using different paths
# python pick_demo.py \
#     --scores_file "path/to/your/scores.txt" \
#     --wer_file "path/to/your/wer_file.txt" \
#     --source_dir "path/to/source/audio/files/" \
#     --target_dir "path/to/target/directory/" \
#     --output_list "selected_wav_files.txt"
