from time import time
from typing import Dict, Optional, Tuple

import torch as t
from einops import einsum
from torch.nn.init import kaiming_uniform_
import math

# Copied from Subnetwork Probing paper: https://github.com/stevenxcao/subnetwork-probing
mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3
p = (mask_p - left) / (right - left)
init_mask_val = math.log(p / (1 - p))

def sample_hard_concrete(mask: t.Tensor) -> t.Tensor:
    """
    Sample from the hard concrete distribution
    ([Louizos et al., 2017](https://arxiv.org/abs/1712.01312)).

    Args:
        mask: The mask whose values parameterize the distribution.

    Returns:
        A sample for each element in the mask for each batch element. The returned
        tensor has shape `(batch_size, *mask.shape)`.
    """
    u = t.zeros_like(mask).uniform_().clamp(0.0001, 0.9999)
    s = t.sigmoid((u.log() - (1 - u).log() + mask) / temp)
    s_bar = s * (right - left) + left
    return s_bar.clamp(min=0.0, max=1.0)

class BinaryAutoencoder(t.nn.Module):
    """
    Binary Autoencoder

    Implements:
        latents = (encoder(x - dec_bias) + enc_bias) > 0 ? 1 : 0
        recons = decoder(latents) + dec_bias
    """

    def __init__(
        self,
        n_latents: int,
        n_inputs: int,
        decoder_bias: Optional[t.Tensor] = None
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the input (e.g residual stream, MLP neurons)
        """
        super().__init__()
        self.init_params(n_latents, n_inputs, decoder_bias)
        self.reset_activated_latents()

    def init_params(
        self,
        n_latents: int,
        n_inputs: int,
        decoder_bias: Optional[t.Tensor] = None,
    ) -> None:
        self.n_latents: int = n_latents
        self.n_inputs: int = n_inputs
        self.dec_bias = t.nn.Parameter(decoder_bias)
        self.enc_bias = t.nn.Parameter(t.zeros([n_latents]))
        self.feature_mags = t.nn.Parameter(t.ones([n_latents]))
        self.encode_weight = t.nn.Parameter(t.zeros([n_latents, n_inputs]))
        self.decode_weight = t.nn.Parameter(t.zeros([n_inputs, n_latents]))
        [kaiming_uniform_(w) for w in [self.encode_weight]]
        self.decode_weight.data = self.encode_weight.data.T
        self.decode_weight.data /= self.decode_weight.data.norm(dim=0)

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
        cls, state_dict: Dict[str, t.Tensor]
    ) -> "BinaryAutoencoder":
        n_latents, n_inputs = state_dict["encode_weight"].shape
        autoencoder = cls(n_latents, n_inputs)
        autoencoder.load_state_dict(state_dict, strict=True, assign=True)
        autoencoder.reset_activated_latents()
        return autoencoder

    def encode(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: input data (shape: [..., [seq], n_inputs])
        :return: autoencoder latents (shape: [..., [seq], n_latents])
        """
        encoded = einsum(x - self.dec_bias, self.encode_weight, "... d, ... l d -> ... l")
        latents_pre_act = encoded + self.enc_bias
        return latents_pre_act

    def decode(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: autoencoder x (shape: [..., n_latents])
        :return: reconstructed data (shape: [..., n_inputs])
        """
        ein_str = "... l, d l -> ... d"
        return einsum(x, self.decode_weight, ein_str) + self.dec_bias

    def forward(self, x: t.Tensor, sample: bool = False) -> Tuple[t.Tensor, ...]:
        """
        :param x: input data (shape: [..., n_inputs])
        :return:  reconstructed data (shape: [..., n_inputs])
        """
        latent_pre_acts = self.encode(x)
        if sample:
            latents = sample_hard_concrete(latent_pre_acts)
        else:
            latents = (latent_pre_acts > 0).float()
        latents = latents * self.feature_mags
        self.latent_total_act += latents.sum_to_size(self.latent_total_act.shape)
        recons = self.decode(latents)
        return recons, latents, latent_pre_acts, self.feature_mags
