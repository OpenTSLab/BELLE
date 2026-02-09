import os
import sys
import math
import abc
import subprocess
from argparse import ArgumentParser

import torch
from collections.abc import Sequence
from typing import Any


class MultiGPUWorker(abc.ABC):
    def __init__(self, ):
        self.parser = ArgumentParser()
        self.launch_arg_keys = [
            "gpu_subprocess", "gpu_id", "start_idx", "end_idx", "num_gpus", "processes_per_gpu"
        ]

    def add_launch_args(self):
        self.parser.add_argument(
            '--gpu_subprocess',
            action='store_true',
            help="Run as a subprocess on each GPU"
        )
        self.parser.add_argument(
            '--gpu_id',
            type=int,
            default=0,
            help="GPU ID to use for this worker"
        )
        self.parser.add_argument(
            '--start_idx',
            type=int,
            default=0,
            help="Start index for item processing"
        )
        self.parser.add_argument(
            '--end_idx',
            type=int,
            default=-1,
            help="End index for item processing"
        )
        self.parser.add_argument(
            '--num_gpus',
            type=int,
            default=None,
            help="Total number of GPUs available"
        )
        self.parser.add_argument(
            '--processes_per_gpu',
            type=int,
            default=1,
            help="Number of processes to launch per GPU"
        )

    @abc.abstractmethod
    def add_running_args(self):
        pass

    @abc.abstractmethod
    def load_item_list(self, args) -> Sequence[Any]:
        pass

    @abc.abstractmethod
    def process_item_list(self, items: Sequence[Any], args):
        pass

    def run(self):
        self.add_launch_args()
        self.add_running_args()
        args = self.parser.parse_args()
        if args.gpu_subprocess:
            self.run_worker(args)
        else:
            self.distribute_across_gpus(args)

    def distribute_across_gpus(self, args):
        all_items = self.load_item_list(args)
        num_gpus = args.num_gpus or torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available. Exiting.")
            return
        
        processes_per_gpu = args.processes_per_gpu
        total_processes = num_gpus * processes_per_gpu
        items_per_process = math.ceil(len(all_items) / total_processes)

        print(f"Distributing {len(all_items)} items to {num_gpus} GPUs with {processes_per_gpu} processes per GPU (total: {total_processes} processes).")

        args_dict = vars(args)
        running_args = []
        for k, v in args_dict.items():
            if k not in self.launch_arg_keys and v is not None:
                running_args.append(f"--{k}")
                running_args.append(str(v))

        processes = []
        process_id = 0
        for gpu_id in range(num_gpus):
            for process_idx in range(processes_per_gpu):
                start = process_id * items_per_process
                end = min((process_id + 1) * items_per_process, len(all_items))
                
                # If there are no more items to process, break
                if start >= len(all_items):
                    break

                subprocess_args = [
                    sys.executable,
                    sys.argv[0],
                    "--gpu_subprocess",
                    "--gpu_id",
                    str(gpu_id),
                    "--start_idx",
                    str(start),
                    "--end_idx",
                    str(end),
                ] + running_args

                print(f"Running GPU-{gpu_id} process-{process_idx} subprocess for range {start}:{end}")
                process = subprocess.Popen(subprocess_args)
                processes.append(process)
                process_id += 1
            
            # If there are no more items to process, break the outer loop
            if start >= len(all_items):
                break

        for i, p in enumerate(processes):
            p.wait()
            if p.returncode != 0:
                print(f"Process {i} failed with return code {p.returncode}.")

    def run_worker(self, args):
        # for cosyvoice vllm, we must set os.environ["CUDA_VISIBLE_DEVICES"], the model can be loaded to the correct GPU. Only setting torch.cuda.set_device(args.gpu_id) will not work, the model will be loaded to the first GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        try:
            # for f5tts, we must set torch.cuda.set_device(args.gpu_id), the model can be loaded to the correct GPU. Only setting os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) will not work, the model will be loaded to the first GPU.
            # setting os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) and setting torch.cuda.set_device(args.gpu_id) at the same time will cause f5tts assertion error, getting dtype when loading model, so I fixed it in f5tts code, setting dtype=torch.float16 manually.
            torch.cuda.set_device(args.gpu_id)
        except Exception as e:
            # setting os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) and setting torch.cuda.set_device(args.gpu_id) at the same time will cause cosyvoice vllm an invalid GPU id error, so need to try-except
            pass
        # for PyTorch >= 1.9, `torch.cuda.device_count()` is wrapped with lru cache,
        # so if we set os.environ["CUDA_VISIBLE_DEVICES"] after `import torch`, it will
        # not work until we clear the cache.
        # try:
        #     torch.cuda.device_count.cache_clear()
        # except AttributeError:
        #     pass
        print(f"[GPU {args.gpu_id}] Running worker")

        all_items = self.load_item_list(args)
        subset = all_items[args.start_idx:args.end_idx]
        print(f"[GPU {args.gpu_id}] Processing {len(subset)} items")
        self.process_item_list(subset, args)


class TestWorker(MultiGPUWorker):
    def add_running_args(self):
        self.parser.add_argument(
            '--test_arg',
            type=int,
            default=5,
            help="An example argument for testing"
        )

    def load_item_list(self, args) -> list[str]:
        return [f"item{i}" for i in range(args.test_arg)]

    def process_item_list(self, items: list[str], args):
        for item in items:
            print(
                f"Processing {item} on GPU {os.environ['CUDA_VISIBLE_DEVICES']}"
            )


if __name__ == "__main__":
    worker = TestWorker()
    worker.run()
