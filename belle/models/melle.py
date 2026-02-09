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


class MELLE(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        pos_learn: bool = False,
        norm_first: bool = True,
        sampling_rate: int = 16000,
        tts_models: List[str] = None,
        num_text_tokens: int = 512,
        audio_embeddings: int = 3000,
        text_embeddings: int = 1000,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()
        self.audio_tokenizer = MelTokenizer(sampling_rate=sampling_rate)

        AUDIO_FEATURE_DIM = self.audio_tokenizer.n_mels

        self.text_embedding = TokenEmbedding(d_model, num_text_tokens)  # W_x

        self.audio_embedding = Prenet(AUDIO_FEATURE_DIM, 3, d_model, 0.5)

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
        self.predict_layer_mean_logvar = nn.Linear(
            d_model, AUDIO_FEATURE_DIM * 2
        )
        self.predict_layer_stop = nn.Linear(d_model, 1)

        self.post_sampler = nn.Sequential(
            nn.Linear(AUDIO_FEATURE_DIM, AUDIO_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(AUDIO_FEATURE_DIM, AUDIO_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(AUDIO_FEATURE_DIM, AUDIO_FEATURE_DIM),
        )

        self.postnet = Postnet(AUDIO_FEATURE_DIM, AUDIO_FEATURE_DIM, n_chans=256)

        self.tts_models = tts_models

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
        y, y_lens = audio_features, audio_feature_lens
        kl_loss_weight = kwargs.get("kl_loss_weight", 0)
        flux_loss_weight = kwargs.get("flux_loss_weight", 0)

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
        output = self.predict_layer_mean_logvar(decoder_output)
        mean, logvar = torch.chunk(output, 2, dim=-1)
        var = torch.exp(logvar)
        logstd = logvar / 2
        noise = torch.randn_like(mean)
        std = torch.exp(logstd)
        z = mean + std * noise
        y1 = z + self.post_sampler(z)
        y2 = y1 + self.postnet(y1)
        
        # Predict stop token
        stop = self.predict_layer_stop(decoder_output)
        
        # Compute loss
        mse_loss1 = F.mse_loss(y1[:, :-1], concat_targets[:, 1:], reduction='none').sum(dim=-1)
        l1_loss1 = F.l1_loss(y1[:, :-1], concat_targets[:, 1:], reduction='none').sum(dim=-1)
        mse_loss2 = F.mse_loss(y2[:, :-1], concat_targets[:, 1:], reduction='none').sum(dim=-1)
        l1_loss2 = F.l1_loss(y2[:, :-1], concat_targets[:, 1:], reduction='none').sum(dim=-1)

        l1_loss = ((l1_loss1 + l1_loss2) * concat_is_audio[:, 1:]).sum()
        mse_loss = ((mse_loss1 + mse_loss2) * concat_is_audio[:, 1:]).sum()
        y1_loss = ((l1_loss1 + mse_loss1) * concat_is_audio[:, 1:]).sum()
        y2_loss = ((l1_loss2 + mse_loss2) * concat_is_audio[:, 1:]).sum()
        regression_loss = l1_loss + mse_loss
        
        kl_loss = 0.5 * torch.sum((mean[:, :-1] - concat_targets[:, 1:])**2 + var[:, :-1] - logvar[:, :-1] - 1, dim=-1)
        kl_loss = (kl_loss * concat_is_audio[:, 1:]).sum()
        
        # Compute flux loss
        flux_loss = - F.l1_loss(mean, concat_targets, reduction='none').sum(dim=-1)
        flux_loss = (flux_loss * concat_is_audio).sum()
        
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
        metrics["kl_loss"] = kl_loss
        metrics["flux_loss"] = flux_loss
        metrics["stop_loss"] = stop_loss
        
        # Compute total loss
        total_loss = regression_loss + kl_loss_weight * kl_loss + flux_loss_weight * flux_loss + stop_loss
        
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
            output = self.predict_layer_mean_logvar(logits)
            mean, logvar = torch.chunk(output, 2, dim=-1)
            logstd = logvar / 2
            std = torch.exp(logstd)
            std_global.append(std.mean().item())
            noise = torch.randn_like(mean)
            z = mean + std * noise
            y1 = z + self.post_sampler(z)

            stop = self.predict_layer_stop(logits)
            stop = torch.sigmoid(stop)
            if stop > 0.9 or (y.shape[1] - prefix_len) > x_len * 8:
                if y.shape[1] == prefix_len:
                    print(stop)
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )
                
                print(f"EOS [{prefix_len} -> {y.shape[1]}]")
                break

            y = torch.concat([y, y1.unsqueeze(1)], dim=1)

        mel = y + self.postnet(y)
        mel = mel[:, prefix_len:]

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
    # python -m belle.models.melle