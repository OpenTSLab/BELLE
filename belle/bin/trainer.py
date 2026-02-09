import argparse
import copy
import logging
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Suppress torchaudio deprecation warnings about StreamingMediaDecoder
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import torch
import torch.distributed as dist
import torch.nn as nn
from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    update_averaged_model,
)
from icefall.dist import cleanup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from belle.data import TtsDataModule
from belle.models import add_model_arguments, get_model
from belle.modules.optim import Eden, Eve, ScaledAdam
from belle.modules.scheduler import get_scheduler

LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module

    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/belle_dev",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="ScaledAdam",
        help="The optimizer.",
    )
    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="Eden",
        help="The scheduler.",
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.05, help="The base learning rate."
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.08,
        help="""The ratio of warmup steps to total training steps. It is used
        when --warmup-steps is 0.""",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=10000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train %% save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=10000,
        help="""Run validation if batch_idx %% valid_interval is 0.""",
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=0,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--accumulate-grad-steps",
        type=int,
        default=1,
        help="""update gradient when batch_idx_train %% accumulate_grad_steps == 0.
        """,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Training dtype: float32 bfloat16 float16.",
    )

    parser.add_argument(
        "--train-stage",
        type=int,
        default=0,
        help="for different stages of training",
    )

    parser.add_argument(
        "--visualize",
        type=str2bool,
        default=False,
        help="visualize model results in eval step.",
    )

    parser.add_argument(
        "--oom-check",
        type=str2bool,
        default=True,
        help="perform OOM check on dataloader batches before starting training.",
    )

    parser.add_argument(
        "--exp-name",
        type=str,
        default="belle",
        help="Experiment name for wandb.",
    )

    parser.add_argument(
        "--kl-loss-weight",
        type=float,
        default=0.0,
        help="KL loss weight for training.",
    )

    parser.add_argument(
        "--edl-loss-weight",
        type=float,
        default=1.0,
        help="EDL loss weight for training.",
    )

    parser.add_argument(
        "--flux-loss-weight",
        type=float,
        default=0.0,
        help="Flux loss weight for training.",
    )

    parser.add_argument(
        "--clip",
        type=float,
        default=2.0,
        help="Gradient clipping.",
    )

    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1700,
        help="Number of steps per epoch.",
    )

    parser.add_argument(
        "--loss-weight",
        type=float,
        nargs="+",
        help="Loss weight for each tts model.",
    )
    parser.add_argument(
        "--coef",
        type=float,
        default=1.0,
        help="Coefficient for the edl reg loss.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100,  # 10: debug 100: train
            "reset_interval": 200,
            "valid_interval": 10000,
            # parameters for TTS
            "env_info": get_env_info(),
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    if isinstance(model, DDP):
        raise ValueError("load_checkpoint before DDP")

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    saved_stage = saved_params.get("train_stage", 1)
    if params.train_stage != saved_stage:
        # switch training stage
        params.start_epoch = 1
        params.start_batch = 0

        for key in ["optimizer", "grad_scaler", "sampler", "scheduler"]:
            if key in saved_params:
                saved_params.pop(key)
    else:
        keys = [
            "best_train_epoch",
            "best_valid_epoch",
            "batch_idx_train",
            "best_train_loss",
            "best_valid_loss",
        ]
        for k in keys:
            params[k] = saved_params[k]

        if params.start_batch > 0:
            if "cur_epoch" in saved_params:
                params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )
    
    # Remove the previous epoch checkpoint unless it's a multiple of 25
    prev_epoch = params.cur_epoch - 1
    prev_checkpoint = params.exp_dir / f"epoch-{prev_epoch}.pt"
    if prev_checkpoint.exists():
        if params.num_epochs >= 100:
            keep_interval = 25
        elif params.num_epochs >= 50:
            keep_interval = 10
        elif params.num_epochs >= 20:
            keep_interval = 5
        else:
            keep_interval = 4
        if prev_epoch % keep_interval != 0:
            prev_checkpoint.unlink()
            logging.info(f"Removed checkpoint: {prev_checkpoint}")


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute transducer loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = (
        model.device
        if isinstance(model, DDP)
        else next(model.parameters()).device
    )
    # at entry, TextTokens is (N, P)
    text_tokens = batch["text_tokens"].to(device)
    text_tokens_lens = batch["text_tokens_lens"].to(device)
    assert text_tokens.ndim == 2

    audio = batch["audio"].to(device)
    audio_lens = batch["audio_lens"].to(device)
    assert audio.ndim == 2

    tts_audios = {}
    tts_models = params.tts_models
    if tts_models and tts_models !=[""]:
        for tts_name in tts_models:
            tts_audios[f"audio_{tts_name}"] = batch[f"audio_{tts_name}"].to(device)
            tts_audios[f"audio_{tts_name}_lens"] = batch[f"audio_{tts_name}_lens"].to(device)

    # Prepare prompt data if stream_mode is enabled
    prompt_kwargs = {}
    if hasattr(params, 'stream_mode') and params.stream_mode:
        if "prompt_text_tokens" in batch and "prompt_audio" in batch:
            prompt_kwargs["prompt_text"] = batch["prompt_text_tokens"].to(device)
            prompt_kwargs["prompt_text_lens"] = batch["prompt_text_tokens_lens"].to(device)
            prompt_kwargs["prompt_audio"] = batch["prompt_audio"].to(device)
            prompt_kwargs["prompt_audio_lens"] = batch["prompt_audio_lens"].to(device)

    kl_loss_weight = 0.0
    edl_loss_weight = params.edl_loss_weight
    flux_loss_weight = params.flux_loss_weight
    if params.batch_idx_train > int(params.num_epochs * params.steps_per_epoch / 40):
        kl_loss_weight = params.kl_loss_weight

    with torch.set_grad_enabled(is_training):
        predicts, loss, metrics = model(
            x=text_tokens,
            x_lens=text_tokens_lens,
            y=audio,
            y_lens=audio_lens,
            train_stage=params.train_stage,
            kl_loss_weight=kl_loss_weight,
            edl_loss_weight=edl_loss_weight,
            flux_loss_weight=flux_loss_weight,
            **tts_audios,
            **prompt_kwargs,
            loss_weight=params.loss_weight,
            coef=params.coef,
        )

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = audio_lens.sum().item() // 256
        info["utterances"] = text_tokens.size(0)

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    for metric in metrics:
        info[metric] = metrics[metric].detach().cpu().item()
    del metrics

    return predicts, loss, info


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    rng: random.Random,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    log_writer = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      rng:
        Random for selecting.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      log_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()
    tot_loss = MetricsTracker()
    iter_dl = iter(train_dl)

    dtype, enabled = torch.float32, False
    if params.dtype in ["bfloat16", "bf16"]:
        dtype, enabled = torch.bfloat16, True
    elif params.dtype in ["float16", "fp16"]:
        dtype, enabled = torch.float16, True

    batch_idx = 0
    while True:
        try:
            batch = next(iter_dl)
        except StopIteration:
            logging.info("Reaches end of dataloader.")
            break

        batch_idx += 1

        params.batch_idx_train += 1
        batch_size = len(batch["text"])

        try:
            with torch.amp.autocast('cuda', dtype=dtype, enabled=enabled):
                _, loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (
                tot_loss * (1 - 1 / params.reset_interval)
            ) + loss_info * (1 / params.reset_interval)

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.

            scaler.scale(loss).backward()
            if params.batch_idx_train >= params.accumulate_grad_steps:
                if (
                    params.batch_idx_train % params.accumulate_grad_steps
                    == 0
                ):
                    if params.optimizer_name not in ["ScaledAdam", "Eve"]:
                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)
                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    for k in range(params.accumulate_grad_steps):
                        if isinstance(scheduler, Eden):
                            scheduler.step_batch(params.batch_idx_train)
                        else:
                            scheduler.step()

            set_batch_count(model, params.batch_idx_train)
        except:  # noqa
            display_and_save_batch(batch, params=params)
            raise

        if params.average_period > 0:
            if (
                params.batch_idx_train > 0
                and params.batch_idx_train % params.average_period == 0
            ):
                # Perform Operation in rank 0
                if rank == 0:
                    update_averaged_model(
                        params=params,
                        model_cur=model,
                        model_avg=model_avg,
                    )
             
        if batch_idx % 100 == 0 and params.dtype in ["float16", "fp16"]:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()
            if cur_grad_scale < 1.0 or (
                cur_grad_scale < 8.0 and batch_idx % 400 == 0
            ):
                scaler.update(cur_grad_scale * 2.0)

            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            cur_grad_scale = (
                scaler._scale.item()
                if params.dtype in ["float16", "fp16"]
                else 1.0
            )

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, train_loss[{loss_info}], "
                f"tot_loss[{tot_loss}], "
                f"batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
                + (
                    f", grad_scale: {cur_grad_scale}"
                    if params.dtype in ["float16", "fp16"]
                    else ""
                )
            )

            if log_writer is not None:
                log_writer.log({"train/learning_rate": cur_lr}, step=params.batch_idx_train)
                for k, v in loss_info.norm_items():
                    log_writer.log({f"train/current_{k}": v}, step=params.batch_idx_train)
                for k, v in tot_loss.norm_items():
                    log_writer.log({f"train/tot_{k}": v}, step=params.batch_idx_train)

                if params.dtype in ["float16", "fp16"]:
                    log_writer.log({"train/grad_scale": cur_grad_scale}, step=params.batch_idx_train)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def init_distributed_mode(rank, local_rank, world_size):
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        device_id=local_rank
    )
    dist.barrier()


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    rng = random.Random(params.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size > 1:
        init_distributed_mode(rank, local_rank, world_size)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        log_writer = wandb.init(
            project="BELLE",
            name=params.exp_name,
            job_type='train',
            mode="offline",
        )
    else:
        log_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    with open(f"{params.exp_dir}/model.txt", "w") as f:
        print(model)
        print(model, file=f)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0 and params.average_period > 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if params.train_stage:
        _model = model.module if isinstance(model, DDP) else model
        model_parameters = _model.stage_parameters(params.train_stage)
    else:
        model_parameters = model.parameters()

    if params.optimizer_name == "ScaledAdam":
        parameters_names = []
        if params.train_stage:  # != 0
            _model = model.module if isinstance(model, DDP) else model
            parameters_names.append(
                [
                    name_param_pair[0]
                    for name_param_pair in _model.stage_named_parameters(
                        params.train_stage
                    )
                ]
            )
        else:
            parameters_names.append(
                [
                    name_param_pair[0]
                    for name_param_pair in model.named_parameters()
                ]
            )

        optimizer = ScaledAdam(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            clipping_scale=params.clip,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
    elif params.optimizer_name == "Eve":
        optimizer = Eve(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.98),
            target_rms=0.1,
        )
    elif params.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            weight_decay=1e-2,
            eps=1e-8,
        )
    elif params.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        raise NotImplementedError()

    scheduler = get_scheduler(params, optimizer)
    optimizer.zero_grad()

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.inf_check:
        register_inf_check_hooks(model)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    dataset = TtsDataModule(args)
    train_cuts = dataset.train_cuts()

    train_dl = dataset.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    if params.oom_check:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
        )

    scaler = GradScaler('cuda', 
        enabled=(params.dtype in ["fp16", "float16"]), init_scale=1.0
    )
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        if isinstance(scheduler, Eden):
            scheduler.step_epoch(epoch - 1)

        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if log_writer is not None:
            log_writer.log({"train/epoch": epoch}, step=params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            rng=rng,
            scaler=scaler,
            log_writer=log_writer,
            world_size=world_size,
            rank=rank,
        )

        try:
            save_checkpoint(
                params=params,
                model=model,
                model_avg=model_avg,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
        except:  # noqa
            logging.warning("Failed to save checkpoint")

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)

    dtype = torch.float32
    if params.dtype in ["bfloat16", "bf16"]:
        dtype = torch.bfloat16
    elif params.dtype in ["float16", "fp16"]:
        dtype = torch.float16

    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.amp.autocast('cuda', dtype=dtype):
                _, loss, _ = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ["RANK"])
    run(rank=rank, world_size=world_size, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
