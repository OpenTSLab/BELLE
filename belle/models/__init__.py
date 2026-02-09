import argparse

import torch.nn as nn
from icefall.utils import AttributeDict, str2bool

from .melle import MELLE
from .belle import BELLE
from .belle_stream import BELLE_Stream


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-name",
        type=str,
        default="BELLE",
        help="BELLE, MELLE, BELLE_Stream.",
    )
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=1024,
        help="Embedding dimension in the decoder model.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=16,
        help="Number of attention heads in the Decoder layers.",
    )
    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=12,
        help="Number of Decoder layers.",
    )
    parser.add_argument(
        "--norm-first",
        type=str2bool,
        default=True,
        help="Pre or Post Normalization.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="""Audio sampling rate.""",
    )
    parser.add_argument(
        "--pos-learn",
        type=str2bool,
        default=False,
        help="Learnable positional embeddings.",
    )
    parser.add_argument(
        "--sample-method",
        type=str,
        choices=["gaussian", "studentt", "inversegamma"],
        default="gaussian",
        help="Sampling method for BELLE model.",
    )
    parser.add_argument(
        "--num-text-tokens",
        type=int,
        default=512,
        help="Number of text tokens for MELLE and BELLE models.",
    )
    parser.add_argument(
        "--audio-embeddings",
        type=int,
        default=3000,
        help="Number of audio embeddings for MELLE and BELLE models.",
    )
    parser.add_argument(
        "--text-embeddings",
        type=int,
        default=1500,
        help="Number of text embeddings for MELLE and BELLE models.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=1,
        help="Number of frames to group together as one input.",
    )
    parser.add_argument(
        "--stream-mode",
        type=str2bool,
        default=False,
        help="Whether to use streaming model.",
    )
    parser.add_argument(
        "--text-chunk-size",
        type=int,
        default=20,
        help="Chunk size for text input in streaming model.",
    )
    parser.add_argument(
        "--audio-chunk-size",
        type=int,
        default=60,
        help="Chunk size for audio in streaming model.",
    )

def get_model(params: AttributeDict) -> nn.Module:
    if params.model_name.lower() in ["melle"]:
        model = MELLE(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            pos_learn=params.get("pos_learn", False),
            tts_models=params.get("tts_models", None),
            sampling_rate=params.get("sampling_rate", 16000),
            num_text_tokens=params.get("num_text_tokens", 512),
            audio_embeddings=params.get("audio_embeddings", 3000),
            text_embeddings=params.get("text_embeddings", 1000),
        )
    elif params.model_name.lower() in ["belle"]:
        model = BELLE(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            pos_learn=params.get("pos_learn", False),
            sampling_rate=params.get("sampling_rate", 16000),
            tts_models=params.get("tts_models", None),
            sample_method=params.get("sample_method", "inversegamma"),
            num_text_tokens=params.get("num_text_tokens", 512),
            audio_embeddings=params.get("audio_embeddings", 3000),
            text_embeddings=params.get("text_embeddings", 1000),
            n_frames=params.get("n_frames", 1),
        )
    elif params.model_name.lower() in ["belle_stream"]:
        model = BELLE_Stream(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            pos_learn=params.get("pos_learn", False),
            sampling_rate=params.get("sampling_rate", 16000),
            tts_models=params.get("tts_models", None),
            sample_method=params.get("sample_method", "inversegamma"),
            num_text_tokens=params.get("num_text_tokens", 512),
            audio_embeddings=params.get("audio_embeddings", 3000),
            text_embeddings=params.get("text_embeddings", 1000),
            n_frames=params.get("n_frames", 1),
            stream_mode=params.get("stream_mode", True),
            text_chunk_size=params.get("text_chunk_size", 20),
            audio_chunk_size=params.get("audio_chunk_size", 60),
        )

    return model
