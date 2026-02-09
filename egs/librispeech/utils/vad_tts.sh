cd ~/BELLE/egs/librispeech
source ~/miniconda3/bin/activate belle
export PYTHONPATH=~/BELLE/icefall:$PYTHONPATH

tts=("cosyvoice" "f5tts" "indextts" "xtts" "maskgct" "sparktts")
for t in ${tts[@]}; do
    python utils/vad.py download/librispeech_$t/ download/librispeech_${t}_vad/
done