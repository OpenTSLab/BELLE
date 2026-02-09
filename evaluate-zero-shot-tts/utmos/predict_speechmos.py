import argparse
import pathlib
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchaudio
from speechmos import UTMOS22Strong
import torch

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", required=False, default=None, type=int)
    parser.add_argument("--mode", required=True, choices=["predict_file", "predict_dir"], type=str)
    parser.add_argument("--ckpt_path", required=False, default="../pretrained/evaluation/utmos22_strong_step7459_v1.pt", type=pathlib.Path)
    parser.add_argument("--inp_dir", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--inp_path", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--out_path", required=True, type=pathlib.Path)
    parser.add_argument("--num_workers", required=False, default=0, type=int)
    parser.add_argument("--task", required=True, choices=["wav_c", "wav_p", "wav_pg", "wav_g"], type=str)
    return parser.parse_args()


class Dataset(Dataset):
    def __init__(self, dir_path: pathlib.Path, task: str):
        self.wavlist = list(dir_path.glob(f"*{task}*.wav"))
        self.wavlist = [f for f in self.wavlist if not f.name.endswith("_ref.wav")]
        _, self.sr = torchaudio.load(self.wavlist[0])

    def __len__(self):
        return len(self.wavlist)

    def __getitem__(self, idx):
        fname = self.wavlist[idx]
        wav, _ = torchaudio.load(fname)
        # wav = wav[:, self.sr * 3:]
        sample = {
            "wav": wav,
            "filename": fname.name  # Add filename
        }
        return sample
    
    def collate_fn(self, batch):
        max_len = max([x["wav"].shape[1] for x in batch])
        out = []
        filenames = []  # Store filenames
        # Performing repeat padding
        for t in batch:
            wav = t["wav"]
            filenames.append(t["filename"])  # Collect filenames
            amount_to_pad = max_len - wav.shape[1]
            padding_tensor = wav.repeat(1,1+amount_to_pad//wav.size(1))
            out.append(torch.cat((wav,padding_tensor[:,:amount_to_pad]),dim=1))
        return torch.stack(out, dim=0), filenames  # Return audio data and filenames


def main():
    args = get_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "predict_file":
        assert args.inp_path is not None, "inp_path is required when mode is predict_file."
        assert args.inp_dir is None, "inp_dir should be None."
        assert args.inp_path.exists()
        assert args.inp_path.is_file()
        wav, sr = torchaudio.load(args.inp_path)
        scorer = UTMOS22Strong()
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        scorer.load_state_dict(state_dict)
        scorer.to(device).eval()
        with torch.no_grad():
            score = scorer(wav.to(device), sr)
        with open(args.out_path, "w") as fw:
            fw.write(f"{args.inp_path.name}: {score[0]}")
    else:
        assert args.inp_dir is not None, "inp_dir is required when mode is predict_dir."
        assert args.bs is not None, "bs is required when mode is predict_dir."
        assert args.inp_path is None, "inp_path should be None."
        assert args.inp_dir.exists()
        assert args.inp_dir.is_dir()
        dataset = Dataset(dir_path=args.inp_dir, task=args.task)
        loader = DataLoader(
            dataset,
            batch_size=args.bs,
            collate_fn=dataset.collate_fn,
            shuffle=False,  # Keep order; do not shuffle
            num_workers=args.num_workers)
        sr = dataset.sr
        # scorer = Score(ckpt_path=args.ckpt_path, input_sample_rate=sr, device=device)
        scorer = UTMOS22Strong()
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        scorer.load_state_dict(state_dict)
        scorer.to(device).eval()
        all_scores = []
        all_filenames = []  # Store all filenames
        for batch, filenames in tqdm.tqdm(loader):
            batch = batch.squeeze(1)
            with torch.no_grad():
                scores = scorer(batch.to(device), sr)
            all_scores.extend(scores.tolist())
            all_filenames.extend(filenames)  # Save filenames
        
        # Compute average score
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        print(f"Average score: {avg_score}")
        
        # Create output directory if missing
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort by filename
        sorted_results = sorted(zip(all_filenames, all_scores), key=lambda x: x[0])
        
        # Write average score and per-file scores
        with open(args.out_path, 'w') as fw:
            fw.write(f"Average score: {avg_score}\n")
            # fw.write("Individual scores:\n")
            # Write each audio filename and its score (sorted by filename)
            for filename, score in sorted_results:
                fw.write(f"{filename}: {score}\n")


if __name__ == '__main__':
    main()