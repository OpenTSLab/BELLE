import logging
import os
import time
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict

from belle.data import (
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from belle.data.collation import get_text_token_collater
from belle.models import get_model

def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)

    args = AttributeDict(checkpoint)
    model = get_model(args)
    model_name = args.model_name.lower()

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    return model, args.text_tokens, model_name


@torch.no_grad()
def main():
    text_tokenizer = TextTokenizer(backend="espeak")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)


    exp_names = ["belle_stream-lr5e-4-flux0-epoch60-cuts_train_filter_all_with_prompts-tts_models_cosyvoice_indextts_sparktts_f5tts_xtts_maskgct-loss_weight_0.22_0.13_0.13_0.13_0.13_0.13_0.13-train_stage2-nframes1-textchunk20-audiochunk50"]

    epochs = [2]

    prompt_names = ["8455_210777_000067_000000"]

    for exp_name in exp_names:
        for epoch in epochs:
            ckpt = f"exp/{exp_name}/epoch-{epoch}.pt"
            model, text_tokens, model_name = load_model(ckpt, device)
            text_collater = get_text_token_collater(text_tokens)

            audio_tokenizer = model.audio_tokenizer
            audio_tokenizer.to(device)
            
            for prompt_name in prompt_names:
                output_dir = f"infer/{exp_name}/epoch={epoch}/{prompt_name}"
                text_prompt = f"prompts/{prompt_name}.txt"
                audio_prompt = f"prompts/{prompt_name}.wav"
                text = f"prompts/prompts_new/1320-122617-0010_wav_g.txt"

                if os.path.isfile(text):
                    with open(text) as f:
                        text = f.read().strip()
                logging.info(f"text: {text}")

                Path(output_dir).mkdir(parents=True, exist_ok=True)

                if os.path.isfile(text_prompt):
                    with open(text_prompt) as f:
                        text_prompt = f.read().strip()
                logging.info(f"text_prompt: {text_prompt}")

                encoded_frames = tokenize_audio(audio_tokenizer, audio_prompt, device)

                prompt_text_tokens, prompt_text_tokens_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=text_prompt.strip()
                        )
                    ]
                )
                text_tokens, text_tokens_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=text.strip()
                        )
                    ]
                )

                # synthesis
                try:
                    start_time = time.time()
                    encoded_frames, std = model.inference(
                        text_tokens.to(device),
                        text_tokens_lens.to(device),
                        encoded_frames.to(device),
                        prompt_text_tokens.to(device),
                        prompt_text_tokens_lens.to(device)
                    )
                    end_time = time.time()
                    total_time = end_time - start_time
                    logging.info(f"Inference time: {total_time:.2f} seconds")

                    # Check if encoded_frames is a list; if so, decode each and concatenate
                    if isinstance(encoded_frames, list):
                        logging.info(f"encoded_frames is a list with {len(encoded_frames)} segments")
                        decoded_segments = []
                        for i, frame in enumerate(encoded_frames):
                            segment_samples = audio_tokenizer.decode(frame)
                            decoded_segments.append(segment_samples[0])  # Assume batch_size=1
                            logging.info(f"Decoded segment {i+1}/{len(encoded_frames)}, frame shape: {frame.shape}")
                        # Concatenate all audio segments
                        samples = torch.cat(decoded_segments, dim=-1).unsqueeze(0)  # Add batch dimension back
                        logging.info(f"Concatenated {len(decoded_segments)} audio segments")
                    else:
                        samples = audio_tokenizer.decode(
                            encoded_frames
                        )
                    
                    audio_len = samples.shape[-1] / audio_tokenizer.sampling_rate
                    logging.info(f"Generated audio length: {audio_len:.2f} seconds")
                    RTF = (total_time) / audio_len
                    logging.info(f"Real-time factor: {RTF:.2f}")
                    # store
                    torchaudio.save(
                        f"{output_dir}/{prompt_name}_3.wav", samples[0].cpu(), audio_tokenizer.sampling_rate
                    )
                except SyntaxError as e:
                    logging.error(f"Error during inference: {e}")
                    continue

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
