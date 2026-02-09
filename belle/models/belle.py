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


class BELLE(nn.Module):
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
        """
        super().__init__()

        self.n_frames = n_frames
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
        
        # Truncate to make T divisible by n_frames (discard remainder)
        T_truncated = T - (T % self.n_frames)
        if T_truncated < T:
            audio_features = audio_features[:, :T_truncated, :]
        
        # Reshape from (B, T_truncated, D) to (B, T_truncated//n_frames, D*n_frames)
        reshaped = audio_features.view(B, T_truncated // self.n_frames, D * self.n_frames)
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
        pred_gt, loss_gt, metrics_gt = self._forward(
            x, x_lens, y, y_lens, **kwargs
        )
        if tts_models is None or tts_models == [""]:
            return pred_gt, loss_gt, metrics_gt
        else:
            assert len(loss_weight) == len(tts_models) + 1
            assert sum(loss_weight) == 1.0
            loss_all = [loss_gt]
            metrics_all = [metrics_gt]
            for tts_name in tts_models:
                audio_tts = kwargs[f"audio_{tts_name}"]
                audio_tts_lens = kwargs[f"audio_{tts_name}_lens"]
                pred_tts, loss_tts, metrics_tts = self._forward(
                    x, x_lens, audio_tts, audio_tts_lens, **kwargs
                )
                loss_all.append(loss_tts)
                metrics_all.append(metrics_tts)

            loss = 0
            keys = metrics_gt.keys()
            metrics = {key: 0 for key in keys}
            for i in range(len(loss_all)):
                loss += loss_weight[i] * loss_all[i]
                for key in keys:
                    metrics[key] += loss_weight[i] * metrics_all[i][key]
            return pred_gt, loss, metrics

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
        """
        audio_features, audio_feature_lens = self.audio_tokenizer.encode(y, y_lens)
        # audio_features: B x T x D
        # audio_feature_lens: B
        
        if self.n_frames > 1:
            audio_features = self._reshape_audio_for_n_frames(audio_features)
            # Update lengths for reshaped features (truncated length)
            audio_feature_lens = audio_feature_lens - (audio_feature_lens % self.n_frames)
            audio_feature_lens = audio_feature_lens // self.n_frames

        y, y_lens = audio_features, audio_feature_lens
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
        x_emb = self.text_position(x_emb)
        
        # Prepare audio embeddings
        y_emb = self.audio_embedding(y)
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
        stop_targets = make_pad_mask(concat_lens, max_len=max_seq_len).to(device).type(torch.float32)
        stop_targets[torch.arange(batch_size), concat_lens - 1] = 1
        stop_targets_weight = stop_targets
        stop_targets_weight = 1 - stop_targets_weight
        stop_targets_weight[torch.arange(batch_size), concat_lens - 1] = 500

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
        
        # Initialize audio position counter and embeddings cache for acceleration
        audio_position_counter = 0
        y_emb_cache = None
        y_pos_cache = None

        while True:
            # Accelerate by computing positions only for newly added y
            if audio_position_counter == 0:
                # First iteration: compute all audio embeddings and positions
                y_emb_cache = self.audio_embedding(y)
                y_pos_cache = self.audio_position(y_emb_cache)
                audio_position_counter = y.shape[1]
            else:
                # Subsequent iterations: only compute embeddings and position for the new audio frame
                new_y = y[:, -1:, :]  # Get the last frame
                new_y_emb = self.audio_embedding(new_y)
                new_audio_pos_indices = torch.tensor([[audio_position_counter]], device=device).long()
                new_audio_pos = self.audio_position(new_y_emb, positions=new_audio_pos_indices)
                
                # Update caches
                y_pos_cache = torch.concat([y_pos_cache, new_audio_pos], dim=1)
                audio_position_counter += 1
            
            xy_pos = torch.concat([x, y_pos_cache], dim=1)

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
            std_global.append(std.mean().item())
            y1 = z + self.post_sampler(z)

            stop = self.predict_layer_stop(logits)
            stop = torch.sigmoid(stop)
            if stop > 0.9 or (y.shape[1] - prefix_len) > x_len * 8 / self.n_frames:
            # if (y.shape[1] - prefix_len) > x_len * 8 / self.n_frames:
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
    # python -m belle.models.belle