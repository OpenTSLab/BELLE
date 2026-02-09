import argparse
import glob
import os
import random
import warnings
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist

# Suppress torchaudio deprecation warnings about StreamingMediaDecoder
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


task_dict = {
    "cont": {
        "task_name": "wav_c",
        "task_dir": {
            "librispeech-test-clean": "evalsets/librispeech-test-clean/exp_base_pl3_r3",
        },
    },
    "cross": {
        "task_name": "wav_pg",
        "task_dir": {
            "librispeech-test-clean": "evalsets/librispeech-test-clean/exp_aligned_pl3_r3",
        },
    },
}


def read_text_from_path(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_txt(path: str, content: str) -> None:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def fix_random_seed(SEED=49):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_unique_key():
    now = datetime.now()
    unique_key = now.strftime("%Y%m%d%H%M%S%f")
    return unique_key


class InferDataset(Dataset):
    def __init__(self, task_key, dataset_key="en"):
        super().__init__()
        assert task_key in ["cont", "cross"]
        task_name, task_dir = (
            task_dict[task_key]["task_name"],
            task_dict[task_key]["task_dir"][dataset_key],
        )
        assert task_name in ["wav_c", "wav_pg"]

        fps = glob.glob(os.path.join(task_dir, f"*_{task_name}_*.wav"))
        fids = set([os.path.basename(x).split("_wav_")[0] for x in fps])
        fids = sorted(fids)
        assert len(fps) % len(fids) == 0
        num_trial = len(fps) // len(fids)

        # Preload dataset
        self.prompt_wav_path = []
        self.prompt_text = []
        self.target_text = []
        for fid in tqdm(fids, desc="preload dataset ..."):
            for i in range(num_trial):
                self.prompt_wav_path.append(
                    os.path.join(task_dir, f"{fid}_{task_name}_{i}.wav")
                )
                prompt_text = read_text_from_path(
                    os.path.join(task_dir, f"{fid}_{task_name}_{i}.txt")
                ).lower()
                self.prompt_text.append(
                    prompt_text
                    if task_name == "wav_pg"
                    else ""
                )  # prompt_text is included in text in continuation task
                ori_target_text = read_text_from_path(
                    os.path.join(task_dir, f"{fid}_wav_g.txt")
                ).lower()
                self.target_text.append(ori_target_text)

        assert len(self.prompt_text) == len(self.target_text)

    def __len__(self):
        return len(self.prompt_wav_path)

    def __getitem__(self, idx):
        meta_data = torchaudio.info(self.prompt_wav_path[idx])
        wav_len = meta_data.num_frames
        sr = meta_data.sample_rate

        sample = {
            "text": self.target_text[idx],
            "prompt_text": self.prompt_text[idx],
            "prompt_wav_len": wav_len,
            "prompt_wav_sr": sr,
            "prompt_wav_path": self.prompt_wav_path[idx],
        }
        return sample

    def to_device(self, batch, device):
        batch["prompt_wav_len"] = batch["prompt_wav_len"].to(device)
        return batch


class InferModel():

    def __init__(self, model_name, backend, ckpt=None, local_rank=0, dataset_name="librispeech"):
        super().__init__()
        self.model_name = model_name
        if model_name == "yourtts":
            from src.evaluate_zero_shot_tts.models.yourtts_inference import YourTTSModel
            self.model = YourTTSModel()
        elif model_name == "valle_lifeiteng":
            from src.evaluate_zero_shot_tts.models.valle_lifeiteng_inference import ValleLifeitengModel
            self.model = ValleLifeitengModel()
        elif model_name == "melle":
            from src.evaluate_zero_shot_tts.models.melle_inference import MelleModel
            self.model = MelleModel(ckpt, backend, local_rank, dataset_name)
        elif model_name == "f5tts":
            from src.evaluate_zero_shot_tts.models.f5tts_inference import F5TTSModel
            self.model = F5TTSModel(local_rank).to(f"cuda:{local_rank}")
        elif model_name == "maskgct":
            from src.evaluate_zero_shot_tts.models.maskgct_inference import MaskGCT
            self.model = MaskGCT(local_rank).to(f"cuda:{local_rank}")
        else:
            raise NotImplementedError()

    def __call__(self, text, prompt_wav_path, task_key, prompt_text=None):
        wav, wav_p, sr, std = self.model.inference(text, prompt_wav_path, task_key, prompt_text)
        if wav_p is not None:
            assert isinstance(wav_p, torch.Tensor) and len(wav_p.shape) == 2
        return wav, wav_p, sr, std


def init_distributed_mode(rank, local_rank, world_size):
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()


def main(args):
    # prepare running
    fix_random_seed(args.seed)

    # setup distributed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    init_distributed_mode(rank, local_rank, world_size)

    # set output directory
    assert args.dataset_key in task_dict[args.task_key]["task_dir"]
    exp_name = args.ckpt.split("/")[-2]
    ckpt_name = args.ckpt.split("/")[-1].replace(".pt", "")
    output_dir = f"{args.output_dir}/{args.dataset_key}/{os.path.basename(task_dict[args.task_key]['task_dir'][args.dataset_key])}/{args.model_name}/{args.task_key}/{exp_name}/{ckpt_name}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"output_dir: {output_dir}\n")

    # data loader
    infer_dataset = InferDataset(args.task_key, args.dataset_key)

    # distributed dataloader
    sampler = DistributedSampler(
        infer_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    infer_dataloader = DataLoader(
        infer_dataset,
        batch_size=1, # currently, batch inference is not supported.
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = InferModel(args.model_name, args.backend, args.ckpt, local_rank, args.dataset_name)

    std_global = {}

    # prompted generation
    for batch in tqdm(infer_dataloader, desc="sampling ..."):
        batch = infer_dataset.to_device(batch, f"cuda:{local_rank}")

        sampled_wav, prompt_wav_recon, sr, std = model(batch['text'][0], batch['prompt_wav_path'][0], args.task_key, None if args.task_key == "cont" else batch['prompt_text'][0])

        if sampled_wav is not None:
            # save wav
            sampled_wav_path = os.path.basename(batch["prompt_wav_path"][0])
            try:
                torchaudio.save(
                    os.path.join(output_dir, sampled_wav_path),
                    sampled_wav.detach().cpu(),
                    sample_rate=sr,
                )
            except Exception as e:
                sf.write(
                    os.path.join(output_dir, sampled_wav_path),
                    sampled_wav.detach().cpu().numpy(),
                    samplerate=sr,
                )

            # save txt
            sample_txt_path = sampled_wav_path.replace(".wav", ".txt")
            save_txt(
                os.path.join(output_dir, sample_txt_path),
                batch["text"][0],
            )

            # save (reconstructed) prmopt wav & text
            if prompt_wav_recon is not None:
                ref_wav_path = os.path.basename(batch["prompt_wav_path"][0]).replace(
                    ".wav", "_ref.wav"
                )
                torchaudio.save(
                    os.path.join(output_dir, ref_wav_path),
                    prompt_wav_recon.detach().cpu(),
                    sample_rate=sr,
                )
                ref_txt_path = ref_wav_path.replace(".wav", ".txt")
                save_txt(
                    os.path.join(output_dir, ref_txt_path),
                    batch["prompt_text"][0],
                )

            std_global[sampled_wav_path] = std

    # save std
    std_output_dir = f"{args.results_dir}/{args.dataset_key}/{args.model_name}/{os.path.basename(task_dict[args.task_key]['task_dir'][args.dataset_key])}/{args.task_key}/{exp_name}/{ckpt_name}/"
    os.makedirs(std_output_dir, exist_ok=True)
    
    # Save per-process std results
    with open(os.path.join(std_output_dir, f"std_rank{rank}.txt"), "w") as f:
        for k, v in sorted(std_global.items()):
            f.write(f"{k}: {v}\n")
    
    # Wait for all processes to finish saving their results
    dist.barrier()
    
    # Only rank 0 merges all results
    if rank == 0:
        all_stds = {}
        # Collect all process results
        for r in range(world_size):
            per_rank_file = os.path.join(std_output_dir, f"std_rank{r}.txt")
            if os.path.exists(per_rank_file):
                with open(per_rank_file, "r") as f:
                    for line in f:
                        if ": " in line:
                            k, v = line.strip().split(": ", 1)
                            all_stds[k] = float(v)
        
        # Write the merged and sorted results
        with open(os.path.join(std_output_dir, "std.txt"), "w") as f:
            if all_stds:
                mean_std = np.mean(list(all_stds.values()))
                f.write(f"mean_std: {mean_std}\n")
                for k, v in sorted(all_stds.items()):
                    f.write(f"{k}: {v}\n")
            else:
                f.write("No standard deviation values were collected.\n")
        
        # Optionally clean up temporary files
        for r in range(world_size):
            per_rank_file = os.path.join(std_output_dir, f"std_rank{r}.txt")
            if os.path.exists(per_rank_file):
                os.remove(per_rank_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser.add_argument(
        "--task_key",
        type=str,
        default="cont",
        choices=["cont", "cross"],
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        default="librispeech-test-clean",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="melle",
        choices=["yourtts", "valle_lifeiteng", "melle", "f5tts", "maskgct"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default="espeak")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    main(args)
