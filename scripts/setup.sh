conda create -n belle python=3.10
source ~/miniconda3/bin/activate belle

pip install -r requirements.txt

apt-get install -y espeak-ng

# k2
# find the right version in https://huggingface.co/csukuangfj/k2
# The cuda version in your system can be higher than the cuda version used to compile k2, but not lower.
wget https://huggingface.co/csukuangfj/k2/tree/main/ubuntu-cuda/k2-1.24.4.dev20250807%2Bcuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
pip install k2-1.24.4.dev20250807%2Bcuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

git clone https://github.com/k2-fsa/icefall
pip install -r icefall/requirements.txt

pip install -e .

pip install -r evaluate-zero-shot-tts/requirements_evaluation.txt
pip install -e belle/ParallelWaveGAN

pip install scipy==1.10