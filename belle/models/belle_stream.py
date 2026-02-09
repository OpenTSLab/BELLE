from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import make_pad_mask

from belle.modules.embedding import TokenEmbedding, PositionalEmbedding, Prenet, Postnet
from belle.modules.transformer import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .visualizer import visualize
from ..data import MelTokenizer
from ..modules.edlloss import EvidentialRegression


class BELLE_Stream(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        pos_learn: bool = False,
        norm_first: bool = True,
        sampling_rate: int = 16000,
        tts_models: List[str] = None,
        sample_method: str = "inversegamma",
        num_text_tokens: int = 512,
        audio_embeddings: int = 3000,
        text_embeddings: int = 1000,
        n_frames: int = 1,
        # Stream interleave parameters
        stream_mode: bool = False,
        text_chunk_size: int = 20,
        audio_chunk_size: int = 60,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
          n_frames:
            The number of frames to predict simultaneously (default: 1).
          stream_mode:
            Whether to use streaming interleave format (default: False).
          text_chunk_size:
            Number of text tokens in each chunk for streaming (default: 20).
          audio_chunk_size:
            Number of audio tokens in each chunk for streaming (default: 60).
        """
        super().__init__()

        self.n_frames = n_frames
        self.stream_mode = stream_mode
        self.text_chunk_size = text_chunk_size
        self.audio_chunk_size = audio_chunk_size

        self.audio_tokenizer = MelTokenizer(sampling_rate=sampling_rate)

        AUDIO_FEATURE_DIM = self.audio_tokenizer.n_mels
        # Effective audio feature dimension considering n_frames
        EFFECTIVE_AUDIO_FEATURE_DIM = AUDIO_FEATURE_DIM * n_frames

        self.text_embedding = TokenEmbedding(d_model, num_text_tokens)  # W_x

        self.audio_embedding = Prenet(EFFECTIVE_AUDIO_FEATURE_DIM, 3, d_model, 0.5)

        self.text_position = PositionalEmbedding(
            num_embeddings=text_embeddings,
            embedding_dim=d_model,
            learned=pos_learn,
        )
        self.audio_position = PositionalEmbedding(
            num_embeddings=audio_embeddings,
            embedding_dim=d_model,
            learned=pos_learn,
        )

        self.decoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        self.num_heads = nhead
        self.predict_layer_edl = nn.Linear(
            d_model, EFFECTIVE_AUDIO_FEATURE_DIM * 4
        )
        self.predict_layer_stop = nn.Linear(d_model, 1)

        self.post_sampler = nn.Sequential(
            nn.Linear(EFFECTIVE_AUDIO_FEATURE_DIM, EFFECTIVE_AUDIO_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(EFFECTIVE_AUDIO_FEATURE_DIM, EFFECTIVE_AUDIO_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(EFFECTIVE_AUDIO_FEATURE_DIM, EFFECTIVE_AUDIO_FEATURE_DIM),
        )

        self.postnet = Postnet(EFFECTIVE_AUDIO_FEATURE_DIM, EFFECTIVE_AUDIO_FEATURE_DIM, n_chans=256)
        
        self.tts_models = tts_models
        self.sample_method = sample_method

    def _reshape_audio_for_n_frames(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Reshape audio features to support n_frames prediction.
        
        Args:
            audio_features: (B, T, D) - Original audio features
            
        Returns:
            reshaped_features: (B, T//n_frames, D*n_frames) - Reshaped for n_frames prediction
        """
        B, T, D = audio_features.shape
        
        # Pad to make T divisible by n_frames
        if T % self.n_frames != 0:
            pad_len = self.n_frames - (T % self.n_frames)
            padding = torch.zeros(B, pad_len, D, device=audio_features.device, dtype=audio_features.dtype)
            audio_features = torch.cat([audio_features, padding], dim=1)
            T = T + pad_len
        
        # Reshape from (B, T, D) to (B, T//n_frames, D*n_frames)
        reshaped = audio_features.view(B, T // self.n_frames, D * self.n_frames)
        return reshaped

    def _unreshape_audio_from_n_frames(self, reshaped_features: torch.Tensor, original_length: int) -> torch.Tensor:
        """
        Unreshape audio features from n_frames prediction back to original format.
        
        Args:
            reshaped_features: (B, T//n_frames, D*n_frames) - Reshaped features
            original_length: Original sequence length before padding
            
        Returns:
            audio_features: (B, original_length, D) - Original format
        """
        B, T_reshaped, D_total = reshaped_features.shape
        D = D_total // self.n_frames
        
        # Reshape from (B, T//n_frames, D*n_frames) to (B, T, D)
        audio_features = reshaped_features.view(B, T_reshaped * self.n_frames, D)
        
        # Trim to original length
        audio_features = audio_features[:, :original_length, :]
        return audio_features


    def _create_interleaved_sequence_with_prompts(
        self,
        x_emb: torch.Tensor,
        x_lens: torch.Tensor,
        y_emb: torch.Tensor,
        y_lens: torch.Tensor,
        y_targets: torch.Tensor,
        prompt_text_emb: torch.Tensor = None,
        prompt_text_lens: torch.Tensor = None,
        prompt_y_emb: torch.Tensor = None,
        prompt_y_lens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Create interleaved sequence with prompt text, prompt audio, and text+audio chunks.
        Format: [prompt_text] + [prompt_audio] + [text_chunk1 + audio_chunk1] + [text_chunk2 + audio_chunk2] + ...
        
        Args:
            x_emb: Main text embeddings (B, S, D)
            x_lens: Main text lengths (B,)
            y_emb: Main audio embeddings (B, T, D)
            y_lens: Main audio lengths (B,)
            y_targets: Main audio targets (B, T, D_audio)
            prompt_text_emb: Prompt text embeddings (B, P, D) (optional)
            prompt_text_lens: Prompt text lengths (B,) (optional)
            prompt_y_emb: Prompt audio embeddings (B, T_p, D) (optional)
            prompt_y_lens: Prompt audio lengths (B,) (optional)
            
        Returns:
            concat_seqs: Interleaved sequences (B, max_len, D)
            concat_targets: Interleaved targets (B, max_len, D_audio)
            concat_lens: Sequence lengths (B,)
            concat_is_audio: Audio position mask (B, max_len)
            valid_batch_indices: List of valid batch indices
        """
        batch_size = x_emb.size(0)
        device = x_emb.device
        d_model = x_emb.size(-1)
        d_audio = y_targets.size(-1)
        
        valid_batch_indices = []
        interleaved_seqs = []
        interleaved_targets = []
        interleaved_is_audio = []
        
        for batch_idx in range(batch_size):
            x_len = x_lens[batch_idx].item()
            y_len = y_lens[batch_idx].item()
            
            # Filter out invalid samples
            if x_len == 0 or y_len == 0:
                continue
            if y_len < x_len * 0.5 or y_len > x_len * 20:
                continue
            
            valid_batch_indices.append(batch_idx)
            
            seq_parts = []
            target_parts = []
            is_audio_parts = []
            
            text_position_counter = 0
            audio_position_counter = 0
            
            # Phase 1: Add prompt text if provided
            if prompt_text_emb is not None and prompt_text_lens is not None:
                prompt_text_len = prompt_text_lens[batch_idx].item()
                if prompt_text_len > 0:
                    prompt_text_data = prompt_text_emb[batch_idx, :prompt_text_len]
                    # Apply position encoding
                    prompt_text_pos_indices = torch.arange(text_position_counter, text_position_counter + prompt_text_len, device=device).long()
                    prompt_text_pos = self.text_position(prompt_text_data.unsqueeze(0), positions=prompt_text_pos_indices.unsqueeze(0))
                    seq_parts.append(prompt_text_pos[0])
                    target_parts.append(torch.zeros(prompt_text_len, d_audio, device=device))
                    is_audio_parts.append(torch.zeros(prompt_text_len, dtype=torch.bool, device=device))
                    text_position_counter += prompt_text_len
            
            # Phase 2: Add prompt audio if provided
            if prompt_y_emb is not None and prompt_y_lens is not None:
                prompt_audio_len = prompt_y_lens[batch_idx].item()
                if prompt_audio_len > 0:
                    prompt_audio_data = prompt_y_emb[batch_idx, :prompt_audio_len]
                    # Apply position encoding
                    prompt_audio_pos_indices = torch.arange(audio_position_counter, audio_position_counter + prompt_audio_len, device=device).long()
                    prompt_audio_pos = self.audio_position(prompt_audio_data.unsqueeze(0), positions=prompt_audio_pos_indices.unsqueeze(0))
                    seq_parts.append(prompt_audio_pos[0])
                    target_parts.append(torch.zeros(prompt_audio_len, d_audio, device=device))
                    is_audio_parts.append(torch.zeros(prompt_audio_len, dtype=torch.bool, device=device))
                    audio_position_counter += prompt_audio_len
            
            # Phase 3: Interleave main text and audio chunks
            text_pos = 0
            audio_pos = 0
            
            while text_pos < x_len:
                # Add text chunk
                text_end = min(text_pos + self.text_chunk_size, x_len)
                text_chunk_len = text_end - text_pos
                text_chunk = x_emb[batch_idx, text_pos:text_end]
                # Apply position encoding
                text_chunk_pos_indices = torch.arange(text_position_counter, text_position_counter + text_chunk_len, device=device).long()
                text_chunk_pos = self.text_position(text_chunk.unsqueeze(0), positions=text_chunk_pos_indices.unsqueeze(0))
                seq_parts.append(text_chunk_pos[0])
                target_parts.append(torch.zeros(text_chunk_len, d_audio, device=device))
                is_audio_parts.append(torch.zeros(text_chunk_len, dtype=torch.bool, device=device))
                text_pos = text_end
                text_position_counter += text_chunk_len
                
                # Add corresponding audio chunk
                audio_end = min(audio_pos + self.audio_chunk_size, y_len)
                audio_chunk_len = audio_end - audio_pos
                if audio_chunk_len > 0:
                    audio_chunk = y_emb[batch_idx, audio_pos:audio_end]
                    audio_target_chunk = y_targets[batch_idx, audio_pos:audio_end]
                    # Apply position encoding
                    audio_chunk_pos_indices = torch.arange(audio_position_counter, audio_position_counter + audio_chunk_len, device=device).long()
                    audio_chunk_pos = self.audio_position(audio_chunk.unsqueeze(0), positions=audio_chunk_pos_indices.unsqueeze(0))
                    seq_parts.append(audio_chunk_pos[0])
                    target_parts.append(audio_target_chunk)
                    is_audio_parts.append(torch.ones(audio_chunk_len, dtype=torch.bool, device=device))
                    audio_pos = audio_end
                    audio_position_counter += audio_chunk_len
            
            # Add remaining audio if any
            if audio_pos < y_len:
                remaining_audio = y_emb[batch_idx, audio_pos:y_len]
                remaining_audio_target = y_targets[batch_idx, audio_pos:y_len]
                remaining_len = y_len - audio_pos
                # Apply position encoding
                remaining_audio_pos_indices = torch.arange(audio_position_counter, audio_position_counter + remaining_len, device=device).long()
                remaining_audio_pos = self.audio_position(remaining_audio.unsqueeze(0), positions=remaining_audio_pos_indices.unsqueeze(0))
                seq_parts.append(remaining_audio_pos[0])
                target_parts.append(remaining_audio_target)
                is_audio_parts.append(torch.ones(remaining_len, dtype=torch.bool, device=device))
            
            # Concatenate all parts
            interleaved_seq = torch.cat(seq_parts, dim=0)
            interleaved_target = torch.cat(target_parts, dim=0)
            interleaved_mask = torch.cat(is_audio_parts, dim=0)
            
            interleaved_seqs.append(interleaved_seq)
            interleaved_targets.append(interleaved_target)
            interleaved_is_audio.append(interleaved_mask)
        
        if len(valid_batch_indices) == 0:
            # No valid samples in batch
            return None, None, None, None, []
        
        # Pad sequences to same length
        max_len = max(seq.size(0) for seq in interleaved_seqs)
        
        concat_seqs = torch.zeros(len(valid_batch_indices), max_len, d_model, device=device, dtype=x_emb.dtype)
        concat_targets = torch.zeros(len(valid_batch_indices), max_len, d_audio, device=device, dtype=y_targets.dtype)
        concat_is_audio = torch.zeros(len(valid_batch_indices), max_len, dtype=torch.bool, device=device)
        concat_lens = torch.zeros(len(valid_batch_indices), dtype=torch.long, device=device)
        
        for i, (seq, target, mask) in enumerate(zip(interleaved_seqs, interleaved_targets, interleaved_is_audio)):
            seq_len = seq.size(0)
            concat_seqs[i, :seq_len] = seq
            concat_targets[i, :seq_len] = target
            concat_is_audio[i, :seq_len] = mask
            concat_lens[i] = seq_len

        return concat_seqs, concat_targets, concat_lens, concat_is_audio, valid_batch_indices

    def stage_parameters(self, stage: int = 1) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters():
            if not name.startswith("audio_tokenizer"):
                print(f"parameter: {name}")
                yield param

    def stage_named_parameters(
        self, stage: int = 1
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for pair in self.named_parameters():
            if not pair[0].startswith("audio_tokenizer"):
                yield pair

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 2-D tensor (audio) of shape (N, T).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `y`
            before padding.
        """
        tts_models = self.tts_models
        loss_weight = kwargs.get("loss_weight", None)
        
        # Get GT loss
        try:
            pred_gt, loss_gt, metrics_gt = self._forward(
                x, x_lens, y, y_lens, **kwargs
            )
        except Exception as e:
            print(f"GT forward pass failed due to batch filtering: {e}")
            pred_gt, loss_gt, metrics_gt = None, None, None
        
        # Initialize collection of all models (GT + TTS models)
        loss_all = []
        metrics_all = []
        valid_indices = []
        
        # Add GT results if valid
        if loss_gt is not None:
            loss_all.append(loss_gt)
            metrics_all.append(metrics_gt)
            valid_indices.append(0)  # GT index is 0
        
        # Process TTS models if they exist
        if tts_models is not None and tts_models != [""]:
            assert len(loss_weight) == len(tts_models) + 1
            assert sum(loss_weight) == 1.0
            
            for idx, tts_name in enumerate(tts_models):
                try:
                    audio_tts = kwargs[f"audio_{tts_name}"]
                    audio_tts_lens = kwargs[f"audio_{tts_name}_lens"]
                    pred_tts, loss_tts, metrics_tts = self._forward(
                        x, x_lens, audio_tts, audio_tts_lens, **kwargs
                    )
                    loss_all.append(loss_tts)
                    metrics_all.append(metrics_tts)
                    valid_indices.append(idx + 1)  # TTS model indices start from 1
                except Exception as e:
                    print(f"TTS model {tts_name} forward pass failed due to batch filtering: {e}")
                    continue
        
        # Check if we have any valid losses
        if len(loss_all) == 0:
            print("All forward passes failed, using dummy loss")
            device = x.device
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            dummy_metrics = {"dummy": dummy_loss}
            return ((x, None), dummy_loss, dummy_metrics)
        
        # If only GT model exists and it's valid, return directly
        if (tts_models is None or tts_models == [""]) and len(loss_all) == 1:
            return pred_gt, loss_gt, metrics_gt
        
        # Multiple models case: compute weighted average
        if tts_models is not None and tts_models != [""]:
            # Normalize weights for valid losses only
            valid_weights = [loss_weight[i] for i in valid_indices]
            weight_sum = sum(valid_weights)
            if weight_sum > 0:
                normalized_weights = [w / weight_sum for w in valid_weights]
            else:
                normalized_weights = [1.0 / len(valid_weights)] * len(valid_weights)
        else:
            # Only GT model case
            normalized_weights = [1.0]
        
        # Compute weighted loss and metrics
        loss = 0
        keys = metrics_all[0].keys() if metrics_all else []
        metrics = {key: 0 for key in keys}
        
        for i, (current_loss, current_metrics) in enumerate(zip(loss_all, metrics_all)):
            weight = normalized_weights[i]
            loss += weight * current_loss
            for key in keys:
                metrics[key] += weight * current_metrics[key]
        
        total_models = len(tts_models) + 1 if tts_models is not None and tts_models != [""] else 1
        # Only print when some models failed (valid count < total count)
        if len(loss_all) < total_models:
            print(f"Using {len(loss_all)}/{total_models} valid losses with normalized weights: {normalized_weights}")
        return pred_gt if pred_gt is not None else ((x, None), None, None), loss, metrics

    def sample(
        self,
        gamma: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ):
        if self.sample_method == "gaussian":
            noise = torch.randn_like(gamma)
            std = torch.sqrt(beta * (1 + v) / ((alpha - 1) * v))
            z = gamma + std * noise
            return z, std
        elif self.sample_method == "studentt":
            sampler = torch.distributions.StudentT(df=2 * alpha, loc=gamma, scale=(beta * (1 + v) / (alpha * v)))
            z = sampler.rsample()
            return z, torch.sqrt(beta * (1 + v) / ((alpha - 1) * v))
        elif self.sample_method == "inversegamma":
            var_sampler = torch.distributions.InverseGamma(alpha, beta)
            var = var_sampler.rsample()
            std = torch.sqrt(var)

            noise_mean = torch.randn_like(gamma)
            mean = gamma + noise_mean * torch.sqrt(var / v)

            noise = torch.randn_like(gamma)
            z = mean + std * noise
            return z, std
        else:
            raise ValueError(f"Unknown sampling method: {self.sample_method}")

    def _forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 2-D tensor (audio) of shape (N, T).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `y`
            before padding.
          prompt_text:
            A 2-D tensor of shape (N, P). Prompt text tokens (optional).
          prompt_text_lens:
            A 1-D tensor of shape (N,). Length of prompt text (optional).
          prompt_audio:
            A 2-D tensor of shape (N, T_prompt). Prompt audio (optional).
          prompt_audio_lens:
            A 1-D tensor of shape (N,). Length of prompt audio (optional).
        """
        # Extract prompt parameters from kwargs
        prompt_text = kwargs.get("prompt_text", None)
        prompt_text_lens = kwargs.get("prompt_text_lens", None)
        prompt_audio = kwargs.get("prompt_audio", None)
        prompt_audio_lens = kwargs.get("prompt_audio_lens", None)
        
        audio_features, audio_feature_lens = self.audio_tokenizer.encode(y, y_lens)
        # audio_features: B x T x D
        # audio_feature_lens: B
        
        if self.n_frames > 1:
            audio_features = self._reshape_audio_for_n_frames(audio_features)
            # Update lengths for reshaped features
            audio_feature_lens = torch.ceil(audio_feature_lens.float() / self.n_frames).long()

        y, y_lens = audio_features, audio_feature_lens
        
        # Process prompt audio if provided
        prompt_y = None
        prompt_y_lens = None
        if prompt_audio is not None and prompt_audio_lens is not None:
            prompt_audio_features, prompt_audio_feature_lens = self.audio_tokenizer.encode(prompt_audio, prompt_audio_lens)
            if self.n_frames > 1:
                prompt_audio_features = self._reshape_audio_for_n_frames(prompt_audio_features)
                prompt_audio_feature_lens = torch.ceil(prompt_audio_feature_lens.float() / self.n_frames).long()
            prompt_y, prompt_y_lens = prompt_audio_features, prompt_audio_feature_lens
        
        edl_loss_weight = kwargs.get("edl_loss_weight", 0)
        flux_loss_weight = kwargs.get("flux_loss_weight", 0)
        coef = kwargs.get("coef", 1.0)

        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        metrics = {}
        total_loss = 0.0

        # Remove padding: extract real content per batch sequence (unpadding)
        batch_size = x.size(0)
        device = x.device
        
        # Prepare text embeddings
        x_emb = self.text_embedding(x)
        
        # Prepare audio embeddings
        y_emb = self.audio_embedding(y)
        
        if self.stream_mode:
            # Streaming mode: Add prompt support
            # Prepare prompt text embeddings (if provided)
            prompt_text_emb = None
            if prompt_text is not None and prompt_text_lens is not None:
                assert prompt_text.ndim == 2, prompt_text.shape
                assert prompt_text_lens.ndim == 1, prompt_text_lens.shape
                prompt_text_emb = self.text_embedding(prompt_text)
            
            # Prepare prompt audio embeddings (if provided)
            prompt_y_emb = None
            if prompt_y is not None:
                assert prompt_y.ndim == 3, prompt_y.shape
                prompt_y_emb = self.audio_embedding(prompt_y)
            
            # Create interleaved sequence with prompts
            result = self._create_interleaved_sequence_with_prompts(
                x_emb, x_lens, y_emb, y_lens, y,
                prompt_text_emb, prompt_text_lens,
                prompt_y_emb, prompt_y_lens
            )
            if result[0] is None:
                # No valid samples in batch, raise exception for forward to catch
                raise RuntimeError("No valid samples in batch after filtering - all samples filtered out")
            
            concat_seqs, concat_targets, concat_lens, concat_is_audio, valid_batch_indices = result

        else:
            # Original concatenation format (no changes)
            x_emb = self.text_position(x_emb)
            y_emb = self.audio_position(y_emb)
            
            # Create containers for concatenated sequences
            max_seq_len = x_lens.max().item() + y_lens.max().item()
            concat_seqs = torch.zeros((batch_size, max_seq_len, x_emb.size(-1)), device=device)
            concat_targets = torch.zeros((batch_size, max_seq_len, y.size(-1)), device=device)
            concat_lens = torch.zeros((batch_size,), dtype=torch.long, device=device)
            concat_is_audio = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
                
            # Unpad and concatenate for each batch sample
            for i in range(batch_size):
                # Extract real text content
                x_len_i = x_lens[i].item()
                x_data_i = x_emb[i, :x_len_i]
                
                # Extract real audio content
                y_len_i = y_lens[i].item()
                y_data_i = y_emb[i, :y_len_i]
                
                # Compute concatenated length
                concat_len = x_len_i + y_len_i
                concat_lens[i] = concat_len
                
                # Concatenate sequences
                concat_seqs[i, :x_len_i] = x_data_i
                concat_seqs[i, x_len_i:concat_len] = y_data_i
                concat_targets[i, x_len_i:concat_len] = y[i, :y_len_i]
                
                # Mark which positions are audio
                concat_is_audio[i, x_len_i:concat_len] = True

        # prepare stop targets
        max_seq_len = concat_seqs.size(1)
        stop_targets = make_pad_mask(concat_lens, max_len=max_seq_len).to(device).type(torch.float32)
        stop_targets[torch.arange(concat_seqs.size(0)), concat_lens - 1] = 1
        stop_targets_weight = stop_targets
        stop_targets_weight = 1 - stop_targets_weight
        stop_targets_weight[torch.arange(concat_seqs.size(0)), concat_lens - 1] = 500

        # prepare causal mask
        attn_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device), diagonal=1
        )
        
        # Decode with Transformer
        decoder_output = self.decoder(concat_seqs, mask=attn_mask)
        
        # Predict audio features
        edl_output = self.predict_layer_edl(decoder_output)
        edl_loss, gamma, v, alpha, beta = EvidentialRegression(concat_targets[:, 1:], edl_output[:, :-1], coef)
        z, _ = self.sample(gamma, v, alpha, beta)
        y1 = z + self.post_sampler(z)
        y2 = y1 + self.postnet(y1)
        
        # Predict stop token
        stop = self.predict_layer_stop(decoder_output)
        
        # Compute loss
        mse_loss1 = F.mse_loss(y1, concat_targets[:, 1:], reduction='none').sum(dim=-1)
        l1_loss1 = F.l1_loss(y1, concat_targets[:, 1:], reduction='none').sum(dim=-1)
        mse_loss2 = F.mse_loss(y2, concat_targets[:, 1:], reduction='none').sum(dim=-1)
        l1_loss2 = F.l1_loss(y2, concat_targets[:, 1:], reduction='none').sum(dim=-1)

        l1_loss = ((l1_loss1 + l1_loss2) * concat_is_audio[:, 1:]).sum()
        mse_loss = ((mse_loss1 + mse_loss2) * concat_is_audio[:, 1:]).sum()
        y1_loss = ((l1_loss1 + mse_loss1) * concat_is_audio[:, 1:]).sum()
        y2_loss = ((l1_loss2 + mse_loss2) * concat_is_audio[:, 1:]).sum()
        regression_loss = l1_loss + mse_loss
        
        edl_loss = edl_loss.sum(dim=-1)
        edl_loss = (edl_loss * concat_is_audio[:, 1:]).sum()
        
        # Compute flux loss
        flux_loss = - F.l1_loss(gamma, concat_targets[:, :-1], reduction='none').sum(dim=-1)
        flux_loss = (flux_loss * concat_is_audio[:, :-1]).sum()
        
        # Compute stop loss
        stop_loss = F.binary_cross_entropy_with_logits(
            stop.squeeze(-1), stop_targets, weight=stop_targets_weight, reduction="none"
        )
        stop_loss = (stop_loss * concat_is_audio).sum()
        
        # Log metrics
        metrics["regression_loss"] = regression_loss
        metrics["l1_loss"] = l1_loss
        metrics["l2_loss"] = mse_loss
        metrics["y1_loss"] = y1_loss
        metrics["y2_loss"] = y2_loss
        metrics["edl_loss"] = edl_loss
        metrics["flux_loss"] = flux_loss
        metrics["stop_loss"] = stop_loss
        
        # Compute total loss
        total_loss = regression_loss + edl_loss_weight * edl_loss + flux_loss_weight * flux_loss + stop_loss
        
        return ((x, concat_seqs), total_loss, metrics)

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        prompt_text: torch.Tensor = None,
        prompt_text_lens: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[List[torch.Tensor], List[float]]]:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, D).
          prompt_text:
            A 2-D tensor of shape (1, P). Prompt text tokens for streaming mode.
          prompt_text_lens:
            A 1-D tensor of shape (1,). Length of prompt text for streaming mode.
        """
        if self.stream_mode:
            return self.inference_stream(x, x_lens, y, prompt_text, prompt_text_lens)
        
        return self.inference_original(x, x_lens, y)

    def inference_original(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, D).
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        device = x.device
        
        # Reshape y for n_frames if necessary
        if self.n_frames > 1:
            y = self._reshape_audio_for_n_frames(y)

        # NOTE: x has been padded in TextTokenCollater
        x = self.text_embedding(x)
        x = self.text_position(x)

        prefix_len = y.shape[1]
        x_len = x_lens.max()

        std_global = []

        while True:
            y_emb = self.audio_embedding(y)
            y_pos = self.audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)

            xy_len = xy_pos.shape[1]

            attn_mask = torch.triu(
                torch.ones(xy_len, xy_len, dtype=torch.bool, device=device), diagonal=1
            )

            xy_dec = self.decoder(
                xy_pos,
                mask=attn_mask,
            )
            logits = xy_dec[:, -1]
            edl_output = self.predict_layer_edl(logits)
            edl_loss, gamma, v, alpha, beta = EvidentialRegression(y, edl_output)
            z, std = self.sample(gamma, v, alpha, beta)
            # z, std = gamma, torch.tensor(0, dtype=torch.float32)
            std_global.append(std.mean().item())
            y1 = z + self.post_sampler(z)

            stop = self.predict_layer_stop(logits)
            stop = torch.sigmoid(stop)
            if stop > 0.9 or (y.shape[1] - prefix_len) > x_len * 8:
                if y.shape[1] == prefix_len:
                    print(f"stop: {stop}")
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )
                
                print(f"EOS [{prefix_len} -> {y.shape[1]}]")
                break

            y = torch.concat([y, y1.unsqueeze(1)], dim=1)

        mel = y + self.postnet(y)
        mel = mel[:, prefix_len:]
        
        # Unreshape mel back to original frame dimension if necessary
        if self.n_frames > 1:
            # Calculate the original length for the generated part
            generated_len = (mel.shape[1] * self.n_frames)
            mel = self._unreshape_audio_from_n_frames(mel, generated_len)

        return mel, std_global

    def inference_stream(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        prompt_text: torch.Tensor = None,
        prompt_text_lens: torch.Tensor = None,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Streaming inference with interleaved text and audio chunks.
        
        Args:
          x:
            A 2-D tensor of shape (1, S). Main text to synthesize.
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, D). Prompt audio features.
          prompt_text:
            A 2-D tensor of shape (1, P). Prompt text tokens.
          prompt_text_lens:
            A 1-D tensor of shape (1,). Length of prompt text.
            
        Returns:
          audio_chunks: List of audio tensors, one for each chunk plus final remaining audio
          std_global: List of std values
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        device = x.device
        
        # Reshape y for n_frames if necessary
        if self.n_frames > 1:
            y = self._reshape_audio_for_n_frames(y)

        # Prepare prompt text embedding if provided
        if prompt_text is not None and prompt_text_lens is not None:
            assert prompt_text.ndim == 2, prompt_text.shape
            assert prompt_text_lens.ndim == 1, prompt_text_lens.shape
            prompt_text_emb = self.text_embedding(prompt_text)
            prompt_len = prompt_text_lens.max().item()
        else:
            prompt_text_emb = None
            prompt_len = 0

        # Prepare main text embedding  
        x_emb = self.text_embedding(x)
        
        prefix_len = y.shape[1]
        x_len = x_lens.max().item()
        
        std_global = []
        audio_chunks = []  # Store audio chunks
        
        # Phase 1: Initialize with prompt text + prompt audio
        current_seq = []
        text_position_counter = 0  # Text position counter
        audio_position_counter = 0  # Audio position counter
        
        # Add prompt text if provided
        if prompt_text_emb is not None and prompt_len > 0:
            # Create position indices for prompt text
            prompt_text_pos_indices = torch.arange(text_position_counter, text_position_counter + prompt_len, device=device).long()
            prompt_text_pos = self.text_position(prompt_text_emb, positions=prompt_text_pos_indices.unsqueeze(0))
            current_seq.append(prompt_text_pos[0])
            text_position_counter += prompt_len
        
        # Add prompt audio with continuous position encoding
        y_emb = self.audio_embedding(y)
        audio_pos_indices = torch.arange(audio_position_counter, audio_position_counter + y.shape[1], device=device).long()
        y_pos = self.audio_position(y_emb, positions=audio_pos_indices.unsqueeze(0))
        current_seq.append(y_pos[0])
        audio_position_counter += y.shape[1]
        
        # Phase 2: Process main text in chunks
        text_pos = 0
        audio_generated_in_chunk = 0
        
        while text_pos < x_len:
            # Add next text chunk with continuous position encoding
            text_end = min(text_pos + self.text_chunk_size, x_len)
            text_chunk_emb = x_emb[0, text_pos:text_end].unsqueeze(0)
            chunk_len = text_end - text_pos
            
            # Apply position encoding with text position counter
            text_chunk_pos_indices = torch.arange(text_position_counter, text_position_counter + chunk_len, device=device).long()
            text_chunk_pos = self.text_position(text_chunk_emb, positions=text_chunk_pos_indices.unsqueeze(0))
            current_seq.append(text_chunk_pos[0])
            text_position_counter += chunk_len
            text_pos = text_end

            if chunk_len < self.text_chunk_size:
                break
            
            # Track the start of this audio chunk
            chunk_start_idx = y.shape[1]
            
            # Generate audio_chunk_size audio tokens for this text chunk
            for _ in range(self.audio_chunk_size):
                # Concatenate current sequence
                xy_pos = torch.cat(current_seq, dim=0).unsqueeze(0)  # (1, seq_len, d_model)
                xy_len = xy_pos.shape[1]

                attn_mask = torch.triu(
                    torch.ones(xy_len, xy_len, dtype=torch.bool, device=device), diagonal=1
                )

                xy_dec = self.decoder(xy_pos, mask=attn_mask)
                logits = xy_dec[:, -1]
                
                edl_output = self.predict_layer_edl(logits)
                edl_loss, gamma, v, alpha, beta = EvidentialRegression(y[:, -1:], edl_output)
                z, std = self.sample(gamma, v, alpha, beta)
                std_global.append(std.mean().item())
                y1 = z + self.post_sampler(z)

                # Add generated audio token to sequence and y with audio position counter
                new_audio_emb = self.audio_embedding(y1.unsqueeze(0))
                new_audio_pos_indices = torch.tensor([[audio_position_counter]], device=device).long()
                new_audio_pos = self.audio_position(new_audio_emb, positions=new_audio_pos_indices)
                current_seq.append(new_audio_pos[0])
                y = torch.cat([y, y1.unsqueeze(1)], dim=1)
                audio_position_counter += 1
                audio_generated_in_chunk += 1
            
            # Extract the chunk audio and apply postnet
            chunk_audio = y[:, chunk_start_idx:] + self.postnet(y[:, chunk_start_idx:])
            
            # Unreshape chunk audio if necessary
            if self.n_frames > 1:
                chunk_len = chunk_audio.shape[1] * self.n_frames
                chunk_audio = self._unreshape_audio_from_n_frames(chunk_audio, chunk_len)
            
            audio_chunks.append(chunk_audio)
        
        # Phase 3: Generate remaining audio until stop
        # print(f"Phase 3: Generating remaining audio until stop...")
        generated_after_text = 0
        final_audio_start_idx = y.shape[1]  # Track where final audio generation starts
        
        while True:
            # Concatenate current sequence
            xy_pos = torch.cat(current_seq, dim=0).unsqueeze(0)  # (1, seq_len, d_model)
            xy_len = xy_pos.shape[1]

            attn_mask = torch.triu(
                torch.ones(xy_len, xy_len, dtype=torch.bool, device=device), diagonal=1
            )

            xy_dec = self.decoder(xy_pos, mask=attn_mask)
            logits = xy_dec[:, -1]
            
            edl_output = self.predict_layer_edl(logits)
            edl_loss, gamma, v, alpha, beta = EvidentialRegression(y[:, -1:], edl_output)
            z, std = self.sample(gamma, v, alpha, beta)
            std_global.append(std.mean().item())
            y1 = z + self.post_sampler(z)

            stop = self.predict_layer_stop(logits)
            stop = torch.sigmoid(stop)
            
            # Check stop condition
            if stop > 0.9 or (audio_generated_in_chunk + generated_after_text) > x_len * 8:
                # if generated_after_text == 0:
                #     print(f"stop: {stop}")
                
                print(f"Stream EOS [generated after text: {generated_after_text} frames]")
                break

            # Add generated audio token to sequence and y with audio position counter
            new_audio_emb = self.audio_embedding(y1.unsqueeze(0))
            new_audio_pos_indices = torch.tensor([[audio_position_counter]], device=device).long()
            new_audio_pos = self.audio_position(new_audio_emb, positions=new_audio_pos_indices)
            current_seq.append(new_audio_pos[0])
            y = torch.cat([y, y1.unsqueeze(1)], dim=1)
            audio_position_counter += 1
            generated_after_text += 1

        # Generate final remaining audio (not cut into chunks)
        if final_audio_start_idx < y.shape[1]:
            final_audio = y[:, final_audio_start_idx:] + self.postnet(y[:, final_audio_start_idx:])
            
            # Unreshape final audio if necessary
            if self.n_frames > 1:
                final_len = final_audio.shape[1] * self.n_frames
                final_audio = self._unreshape_audio_from_n_frames(final_audio, final_len)
            
            audio_chunks.append(final_audio)

        # print(f"Total generated: {y.shape[1] - prefix_len} frames")
        # print(f"Number of audio chunks: {len(audio_chunks)}")
        return audio_chunks, std_global

    
    def visualize(
        self,
        predicts: Tuple[torch.Tensor],
        batch: Dict[str, Union[List, torch.Tensor]],
        output_dir: str,
        limit: int = 4,
    ) -> None:
        visualize(predicts, batch, output_dir, limit=limit)


if __name__ == "__main__":
    pass
    # python -m belle.models.belle_stream