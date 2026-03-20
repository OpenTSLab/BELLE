import argparse
import pathlib
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchaudio
from speechmos import UTMOS22Strong
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
import gzip
import os

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", required=False, default=None, type=int)
    parser.add_argument("--ckpt_path", required=False, default="../pretrained/evaluation/utmos22_strong_step7459_v1.pt", type=pathlib.Path)
    parser.add_argument("--jsonl_path", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--out_path", required=True, type=pathlib.Path)
    parser.add_argument("--num_workers", required=False, default=0, type=int)
    parser.add_argument("--world_size", required=False, default=1, type=int, help="Number of GPUs to use")
    parser.add_argument("--master_port", required=False, default="12355", type=str, help="Master port for distributed training")
    parser.add_argument("audio_base_dir", type=pathlib.Path, help="Base directory for audio files (used to resolve relative paths in jsonl)", default=pathlib.Path("../egs/librispeech"))
    return parser.parse_args()


def setup_distributed():
    """Set up distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()
    return True, rank, world_size, gpu


class JsonlDataset(Dataset):
    def __init__(self, jsonl_path: pathlib.Path, audio_base_dir: pathlib.Path):
        self.data = []
        self.audio_paths = []
        
        # Read jsonl.gz file
        with gzip.open(jsonl_path, 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
                # Extract audio file path
                audio_source = audio_base_dir / item['recording']['sources'][0]['source']
                self.audio_paths.append(audio_source)
        
        # Get sample rate (from the first audio file)
        if self.audio_paths:
            _, self.sr = torchaudio.load(self.audio_paths[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        wav, _ = torchaudio.load(audio_path)
        sample = {
            "wav": wav,
            "data_idx": idx,
            "audio_path": audio_path
        }
        return sample
    
    def collate_fn(self, batch):
        max_len = max([x["wav"].shape[1] for x in batch])
        out = []
        data_indices = []
        audio_paths = []
        
        # Performing repeat padding
        for t in batch:
            wav = t["wav"]
            data_indices.append(t["data_idx"])
            audio_paths.append(t["audio_path"])
            amount_to_pad = max_len - wav.shape[1]
            padding_tensor = wav.repeat(1,1+amount_to_pad//wav.size(1))
            out.append(torch.cat((wav,padding_tensor[:,:amount_to_pad]),dim=1))
        return torch.stack(out, dim=0), data_indices, audio_paths


def main():
    args = get_arg()

    distributed, rank, world_size, gpu = setup_distributed()

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    assert args.jsonl_path.exists(), f"jsonl file {args.jsonl_path} does not exist"
    assert args.bs is not None, "bs is required for jsonl mode"
    
    dataset = JsonlDataset(jsonl_path=args.jsonl_path, audio_base_dir=args.audio_base_dir)
    
    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None
    
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers)
    sr = dataset.sr
    
    # Initialize model
    scorer = UTMOS22Strong()
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    scorer.load_state_dict(state_dict)
    scorer.to(device).eval()
    
    # Process audio and get scores
    all_scores = []
    all_data_indices = []
    
    progress_bar = tqdm.tqdm(loader) if rank == 0 else loader
    
    for batch, data_indices, audio_paths in progress_bar:
        batch = batch.squeeze(1)
        
        with torch.no_grad():
            scores = scorer(batch.to(device), sr)
        batch_scores = scores.detach().cpu().tolist()
        
        all_scores.extend(batch_scores)
        all_data_indices.extend(data_indices)
    
    # Gather results from all processes
    if world_size > 1:
        # Convert to tensors for gathering
        all_scores_tensor = torch.tensor(all_scores, device=device)
        all_indices_tensor = torch.tensor(all_data_indices, device=device)
        
        # Gather all scores and indices
        gathered_scores = [torch.zeros_like(all_scores_tensor) for _ in range(world_size)]
        gathered_indices = [torch.zeros_like(all_indices_tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered_scores, all_scores_tensor)
        dist.all_gather(gathered_indices, all_indices_tensor)
        
        if rank == 0:
            # Combine results from all processes
            final_scores = []
            final_indices = []
            for scores_tensor, indices_tensor in zip(gathered_scores, gathered_indices):
                final_scores.extend(scores_tensor.cpu().tolist())
                final_indices.extend(indices_tensor.cpu().tolist())
            
            # Add scores to dataset
            for idx, score in zip(final_indices, final_scores):
                dataset.data[idx]['supervisions'][0]['custom']['mos_score'] = score
    else:
        # Single GPU case
        for idx, score in zip(all_data_indices, all_scores):
            dataset.data[idx]['supervisions'][0]['custom']['mos_score'] = score

    # Only main process (rank 0) saves the file
    if rank == 0:
        # Compute average score
        if world_size > 1:
            all_scores, all_data_indices = final_scores, final_indices

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        median_score = sorted(all_scores)[len(all_scores)//2] if all_scores else 0
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 0
        
        print(f"Average MOS score: {avg_score}")
        print(f"Median MOS score: {median_score}")
        print(f"Min MOS score: {min_score}")
        print(f"Max MOS score: {max_score}")
        
        # Create output directory if missing
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save jsonl.gz file with scores
        with gzip.open(args.out_path, 'wt', encoding='utf-8') as f:
            for item in dataset.data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Results saved to {args.out_path}")


if __name__ == '__main__':
    main()