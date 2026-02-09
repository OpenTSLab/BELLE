model_name=melle
kl_loss_weight=0.1
edl_loss_weight=0
flux_loss_weight=0.5
num_epochs=60
lr=5e-4
clip=1.5
scheduler=linear
dataset=librispeech
max_duration=60
steps_per_epoch=7546
## mos
# filter 4.3: 8GPU, step 265

## vad
# cuts train, total steps: 450k
# filter all, all tts models, duration 60, 16GPU: steps 7546, epoch 60, total 450k steps
# filter all, no tts models, duration 480, 8GPU: steps 1486, epoch 300
# combined all, no tts models, duration 480, 8GPU: steps 10209, epoch 45, total 450k steps

optimizer_name=AdamW
pos_learn=True
g2p=espeak
warmup_ratio=0.08
train_stage=1
start_epoch=1
train_cuts_path=data/tokenized/vad_lt_14/cuts_train/filter_all.jsonl.gz
# Extract filename part, remove path and suffix
parent_dir=$(basename "$(dirname "${train_cuts_path}")")
train_cuts_filename="${parent_dir}_$(basename "${train_cuts_path}" .jsonl.gz)"
# tts_models=("")
# loss_weight=(1)
tts_models=("cosyvoice" "indextts" "sparktts" "f5tts" "xtts" "maskgct")
loss_weight=(0.22 0.13 0.13 0.13 0.13 0.13 0.13)

coef=0.5
sample_method=inversegamma
# gaussian, studentt, inversegamma
num_text_tokens=512
audio_embeddings=5000
text_embeddings=2000
text_tokens=data/tokenized/unique_text_tokens.k2symbols
n_frames=1

old_IFS=$IFS
IFS='_'
exp_name=${model_name}-lr${lr}-flux${flux_loss_weight}-epoch${num_epochs}-${train_cuts_filename}-tts_models_${tts_models[*]}-loss_weight_${loss_weight[*]}-train_stage${train_stage}-nframes${n_frames}
IFS=$old_IFS