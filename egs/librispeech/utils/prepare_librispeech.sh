if [ ! -e data/manifests/.librispeech.done ]; then
    lhotse prepare librispeech -j 15 \
      -p dev-clean \
      -p dev-other \
      -p test-clean \
      -p test-other \
      -p train-clean-100 \
      -p train-clean-360 \
      -p train-other-500 "download/LibriSpeech" "data/manifests"
    touch data/manifests/.librispeech.done
fi

dl_dir=$PWD/download
# dataset_parts="-p dev-clean -p test-clean"  # debug
dataset_parts="all"  # all
# audio_extractor="Fbank"  # or Fbank
audio_feats_dir=data/tokenized

mkdir -p ${audio_feats_dir}
if [ ! -e ${audio_feats_dir}/.librispeech.tokenize.done ]; then
    python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" \
        --src-dir "data/manifests/" \
        --output-dir "${audio_feats_dir}" \
        --prefix "librispeech"
fi
touch ${audio_feats_dir}/.librispeech.tokenize.done

lhotse combine \
  ${audio_feats_dir}/librispeech_cuts_train-clean-100.jsonl.gz \
  ${audio_feats_dir}/librispeech_cuts_train-clean-360.jsonl.gz \
  ${audio_feats_dir}/librispeech_cuts_train-other-500.jsonl.gz \
  ${audio_feats_dir}/cuts_train.jsonl.gz

lhotse copy \
  ${audio_feats_dir}/librispeech_cuts_dev-clean.jsonl.gz \
  ${audio_feats_dir}/cuts_dev.jsonl.gz

lhotse copy \
  ${audio_feats_dir}/librispeech_cuts_test-clean.jsonl.gz \
  ${audio_feats_dir}/cuts_test.jsonl.gz