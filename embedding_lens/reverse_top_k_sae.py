import torch as t
from datetime import datetime
from typing import Dict, Final, Optional, Tuple, List

import plotly.express as px
import plotly.graph_objects as go
import torch as t
from einops import einsum
from sympy import N
from torch.nn.functional import cosine_similarity
from torch.nn.init import kaiming_uniform_
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformer_lens import HookedTransformer

from embedding_lens.custom_tqdm import tqdm
from embedding_lens.embeds import get_embeds
from embedding_lens.fast_features import attn_0_derived_features
from embedding_lens.linear_model_sae import no_layernorm_gpt2
from embedding_lens.lr_scheduler import get_scheduler
from embedding_lens.utils import repo_path_to_abs_path
from embedding_lens.visualize import plot_direction_unembeds, plot_word_scores
from tuned_lens import TunedLens

from transformers import AutoModelForCausalLM

class ReverseTopKSparseAutoencoder(t.nn.Module):
    """
    Reverse Top-K Sparse Autoencoder

    Implements:
        latents = TopK(encoder(x - dec_bias))
        recons = decoder(latents) + dec_bias
    """

    def __init__(
        self,
        n_latents: int,
        n_inputs: int,
        batch_size: int,
        zipf_coeffs: Tuple[float, float],
        decoder_bias: Optional[t.Tensor] = None,
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the input (e.g residual stream, MLP neurons)
        :param batch_size: number of samples in the batch for which the autoencoder is being trained
        :param zipf_coeffs: (alpha, beta) for the Zipf distribution
        """
        super().__init__()
        self.init_params(n_latents, n_inputs, batch_size, zipf_coeffs, decoder_bias)
        self.reset_activated_latents()

    def init_params(
        self,
        n_latents: int,
        n_inputs: int,
        batch_size: int,
        zipf_coeffs: Tuple[float, float],
        decoder_bias: Optional[t.Tensor] = None,
    ) -> None:
        self.n_latents: int = n_latents
        self.n_inputs: int = n_inputs
        self.batch_size: int = batch_size
        self.zipf_coeffs: Tuple[float, float] = zipf_coeffs
        if decoder_bias is None:
            decoder_bias = t.zeros(n_inputs, device=DEVICE)
        self.dec_bias = t.nn.Parameter(decoder_bias)
        self.encode_weight = t.nn.Parameter(t.zeros([n_latents, n_inputs]))
        self.decode_weight = t.nn.Parameter(t.zeros([n_inputs, n_latents]))
        [kaiming_uniform_(w) for w in [self.encode_weight, self.decode_weight]]
        self.decode_weight.data /= self.decode_weight.data.norm(dim=0)
        zipf_freq = t.arange(1, n_latents + 1, device=self.dec_bias.device)
        zipf_freq = 1 / ((zipf_freq + zipf_coeffs[1]).pow(zipf_coeffs[0]))
        self.feature_topk_indices = (batch_size * zipf_freq).ceil().long().unsqueeze(0)
        self.feature_topk_indices = self.feature_topk_indices - 1  # 0-indexed
        print(
            "feature_topk",
            self.feature_topk_indices.shape,
            "feature_topk",
            self.feature_topk_indices,
        )

    def reset_activated_latents(
        self, batch_len: Optional[int] = None, seq_len: Optional[int] = None
    ):
        device = self.dec_bias.device
        batch_shape = [] if batch_len is None else [batch_len]
        seq_shape = [] if seq_len is None else [seq_len]
        shape = batch_shape + seq_shape + [self.n_latents]
        self.register_buffer("latent_total_act", t.zeros(shape, device=device), False)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, t.Tensor],
        batch_size: int,
        zipf_coeffs: Tuple[float, float],
    ) -> "ReverseTopKSparseAutoencoder":
        n_latents, n_inputs = state_dict["encode_weight"].shape
        autoencoder = cls(n_latents, n_inputs, batch_size, zipf_coeffs)
        autoencoder.load_state_dict(state_dict, strict=True, assign=True)
        autoencoder.reset_activated_latents()
        return autoencoder

    def encode(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: input data (shape: [..., [seq], n_inputs])
        :return: autoencoder latents (shape: [..., [seq], n_latents])
        """
        x = x - self.dec_bias
        encoded = einsum(
            x / x.norm(dim=-1, keepdim=True),
            self.encode_weight / self.encode_weight.norm(dim=-1, keepdim=True),
            "... d, ... l d -> ... l",
        )
        # encoded = einsum(x, self.encode_weight, "... d, ... l d -> ... l")
        return encoded

    def decode(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: autoencoder x (shape: [..., n_latents])
        :return: reconstructed data (shape: [..., n_inputs])
        """
        ein_str = "... l, d l -> ... d"
        return einsum(x, self.decode_weight, ein_str) + self.dec_bias

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, ...]:
        """
        :param x: input data (shape: [..., n_inputs])
        :return:  reconstructed data (shape: [..., n_inputs])
        """
        assert x.shape[0] == self.batch_size, f"Batch size should be {self.batch_size}"
        latents = self.encode(x)

        # MAKE IT ABSOLUTE VALUE
        # sorting_latents = t.abs(latents)
        sorting_latents = latents

        sorted_values, _ = sorting_latents.sort(dim=0, descending=True)
        # Get the kth values for each latent
        kth_values = sorted_values.gather(dim=0, index=self.feature_topk_indices)
        # Set the values less than the kth value to 0
        mask = sorting_latents >= kth_values.squeeze()
        latents = mask * sorting_latents

        # Soft thresholding
        # sorted_values, _ = t.sort(latents, dim=0, descending=True)
        # kth_values = t.gather(sorted_values, 0, self.feature_topk_indices)
        # latents = t.nn.functional.relu(latents - kth_values)
        # latents = t.where(latents > 0, latents + kth_values, latents)

        # # MAKE IT BINARY
        # non_zero_mask = latents != 0
        # latents = t.where(non_zero_mask, latents / latents, latents)

        self.latent_total_act += latents.sum_to_size(self.latent_total_act.shape)
        recons = self.decode(latents)
        return recons, latents
