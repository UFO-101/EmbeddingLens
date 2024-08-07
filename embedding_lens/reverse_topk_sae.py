#%%
from typing import Dict, Final, Optional, Tuple

import plotly.express as px
import plotly.graph_objects as go
import torch as t
from einops import einsum
from sympy import N
from torch.nn.functional import cosine_similarity
from torch.nn.init import kaiming_uniform_
from transformer_lens import HookedTransformer

from embedding_lens.custom_tqdm import tqdm
from embedding_lens.embeds import get_embeds
from embedding_lens.lr_scheduler import get_scheduler
from embedding_lens.utils import repo_path_to_abs_path
from embedding_lens.visualize import plot_word_scores

MODEL_NAME: Final[str] = "gpt2"
DEVICE = "cuda:2" if t.cuda.is_available() else "cpu"
# DEVICE = "cpu"
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
d_model = model.cfg.d_model
embeds, tok_strs = get_embeds(model, DEVICE, en_only=True)

# %%
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
        decoder_bias = decoder_bias or t.zeros(n_inputs, device=DEVICE)
        self.dec_bias = t.nn.Parameter(decoder_bias)
        self.encode_weight = t.nn.Parameter(t.zeros([n_latents, n_inputs]))
        self.decode_weight = t.nn.Parameter(t.zeros([n_inputs, n_latents]))
        [kaiming_uniform_(w) for w in [self.encode_weight, self.decode_weight]]
        self.decode_weight.data /= self.decode_weight.data.norm(dim=0)
        zipf_freq = t.arange(1, n_latents + 1, device=self.dec_bias.device) 
        zipf_freq = 1 / ((zipf_freq + zipf_coeffs[1]).pow(zipf_coeffs[0]))
        self.feature_topk_indices = (batch_size * zipf_freq).ceil().long().unsqueeze(0)
        self.feature_topk_indices = self.feature_topk_indices - 1 # 0-indexed
        print("feature_topk", self.feature_topk_indices.shape, "feature_topk", self.feature_topk_indices)

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
        encoded = einsum(x, self.encode_weight, "... d, ... l d -> ... l")
        mag_product = x.norm(dim=-1) * self.encode_weight.norm(dim=-1)
        encoded_cosine_sim = encoded / mag_product
        return encoded_cosine_sim
        # return encoded

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
        sorted_values, _ = latents.sort(dim=0, descending=True)
        # Get the kth values for each latent
        kth_values = sorted_values.gather(dim=0, index=self.feature_topk_indices)
        # Set the values less than the kth value to 0
        mask = latents >= kth_values.squeeze()
        latents = latents * mask
        self.latent_total_act += latents.sum_to_size(self.latent_total_act.shape)
        recons = self.decode(latents)
        return recons, latents


def train(
    embeds: t.Tensor,
    N_FEATURES: int,
    d_model: int,
    LR: float,
    N_EPOCHS: int,
    DEVICE: str,
    lr_schedule: bool = False,
    zipf_coeffs: Tuple[float, float] = (1.0, 2.7),
) -> ReverseTopKSparseAutoencoder:
    # assert embeds.shape[0] > N_FEATURES
    sae = ReverseTopKSparseAutoencoder(
        n_latents=N_FEATURES,
        n_inputs=d_model,
        batch_size=embeds.shape[0],
        zipf_coeffs=zipf_coeffs,
        decoder_bias=embeds.mean(0),
    ).to(DEVICE)
    optim = t.optim.Adam(sae.parameters(), lr=LR)
    scheduler = get_scheduler(
        scheduler_name="CosineAnnealingWarmRestarts",
        optimizer=optim,
        training_steps=N_EPOCHS,
        lr = 4e-2,
        warm_up_steps=0,
        decay_steps=0,
        lr_end=0.0,
        num_cycles=10,
    )

    step_history, mse_history, loss_history, lr_history = [], [], [], []
    n_dead_feature_history = []

    step=0
    for epoch in (pbar:=tqdm(range(N_EPOCHS))):
        embed_batch = embeds
        reconstruction, intermediate = sae(embed_batch)
        mses = (reconstruction - embed_batch).pow(2).sum(dim=-1)
        mse_term = mses.mean()
        loss = mse_term

        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"mse {mse_term.item():.2f} loss {loss.item():.2f}")

        step += 1
        if lr_schedule:
            scheduler.step()

        dead_features = intermediate.sum(dim=0) == 0
        n_dead_features = dead_features.sum().item()
        if step % 1 == 0:
            step_history.append(step)
            mse_history.append(mse_term.item())
            loss_history.append(loss.item())
            lr_history.append(optim.param_groups[0]["lr"])
            n_dead_feature_history.append(n_dead_features)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_history, y=loss_history, name="Loss"))
    fig.add_trace(go.Scatter(x=step_history, y=mse_history, name="MSE"))
    fig.add_trace(go.Scatter(x=step_history, y=lr_history, name="LR"))
    fig.add_trace(go.Scatter(x=step_history, y=n_dead_feature_history, name="Dead Features"))
    fig.show()
    return sae

# WHY ARE MSEs SO HIGH?

TRAIN = False
SAVE = False
LOAD = True

N_FEATURES = 1000
N_EPOCHS = 100_000
LR = 1e-2
ZIPF_COEFFS = (1.0, 7.0)


sae_name = f"reverse_topk_zipf_{ZIPF_COEFFS[0]}_a_{ZIPF_COEFFS[1]}_b"
name = f"{sae_name}_{N_FEATURES}_feats_{N_EPOCHS}_epochs_{LR}_lr"
file_path = repo_path_to_abs_path(f"trained_saes/{name}.pth")

if TRAIN:
    sae = train(
        embeds=embeds,
        N_FEATURES=N_FEATURES,
        d_model=d_model,
        LR=LR,
        N_EPOCHS=N_EPOCHS,
        DEVICE=DEVICE,
        lr_schedule=False,
        zipf_coeffs=ZIPF_COEFFS,
    )
if SAVE:
    t.save(sae.state_dict(), file_path)
if LOAD:
    # sae = ReverseTopKSparseAutoencoder(N_FEATURES, d_model, embeds.shape[0], ZIPF_COEFFS)
    # sae.load_state_dict(t.load(file_path, map_location=DEVICE))
    sae = ReverseTopKSparseAutoencoder(20518, d_model, embeds.shape[0], (1.0, 21000))
    sae.to(DEVICE)
    state_dict = {
        # "encode_weight": embeds / (embeds.norm(dim=-1, keepdim=True) ** 2),
        "encode_weight": embeds,
        "decode_weight": embeds.T,
        "dec_bias": t.zeros_like(embeds.mean(0), device=DEVICE),
    }
    sae.load_state_dict(state_dict)

# %%
with t.no_grad():
    print("hi")
    recons, latents = sae(embeds)
    l1s = t.abs(latents).sum(dim=-1)
    print("recons", recons.shape, "latents", latents.shape)
    recons_l2_dists = (recons - embeds).pow(2).sum(dim=-1)
    print("recons_l2_dist", recons_l2_dists.shape)
    normalized_l2_dists = recons_l2_dists / embeds.pow(2).sum(dim=-1)
    print("normalized_l2_dists", normalized_l2_dists.shape)
    cosine_sims = cosine_similarity(recons, embeds, dim=-1)
    print("cosine_sims", cosine_sims.shape)
    non_zero_latents = (latents != 0).sum(dim=-1)
    print("non_zero_latents", non_zero_latents.shape)
    non_zero_occurences = (latents != 0).sum(dim=0)
    print("non_zero_occurences", non_zero_occurences.shape)

# Plot distribution of reconstruction L2 distance
px.histogram(x=recons_l2_dists.cpu().numpy(), nbins=100, title="MSEs").show()
# Plot distribution of normalized L2 distance
px.histogram(x=normalized_l2_dists.cpu().numpy(), nbins=100, title="Normalized L2s").show()
# Plot distribution of cosine similarity
px.histogram(x=cosine_sims.cpu().numpy(), nbins=100, title="Cosine Similarity").show()
# Plot distribution of L1s
px.histogram(x=l1s.cpu().numpy(), nbins=100, title="L1s").show()
# Plot distribution of non-zero latents by token
px.histogram(x=non_zero_latents.cpu().numpy(), nbins=100, title="L0s").show()
# Plot distribution of non_zero tokens by latent
px.histogram(
    x=non_zero_occurences.cpu().numpy(),
    nbins=100,
    title="Number of times features activated"
).show()

#%%
N_FEATURES = 5
with t.no_grad():
    feature_output_logits = embeds @ sae.decode_weight
# For a random sample of features, plot the top activating tokens
for i in range(N_FEATURES):
    feat_idx = int(t.randint(0, sae.n_latents, (1,)).item())
    plot_word_scores(
        latents[:, feat_idx],
        tok_strs,
        title=f"Feature {feat_idx}: Top activating tokens",
    ).show()
    plot_word_scores(
        feature_output_logits[:, feat_idx],
        tok_strs,
        title=f"Feature {feat_idx}: Top output logits",
    ).show()
# %%

N_TOKENS = 5
N_FEATURES_PER_TOKEN = 5
# For a random sample of tokens, show the top activating features
# AND for each of those features, show the top activating tokens
for i in range(N_TOKENS):
    rand_tok_idx = int(t.randint(0, embeds.shape[0], (1,)).item())
    print(f"Token {rand_tok_idx}, '{tok_strs[rand_tok_idx]}'")
    top_activations, top_activating_features = latents[rand_tok_idx].topk(N_FEATURES_PER_TOKEN)
    for rank, (feat_idx, activation) in enumerate(zip(top_activating_features, top_activations)):
        plot_word_scores(
            latents[:, feat_idx],
            tok_strs,
            title=f"Feature {feat_idx}: Top activating tokens",
        ).show()


        
# %%
### ---- OV Feature Extraction ---- ###
def trnsp(x: t.Tensor):
    return x.transpose(-1, -2)

WO = model.blocks[0].attn.W_O
print("WO.shape", WO.shape)
WV = model.blocks[0].attn.W_V
print("WV.shape", WV.shape)

VO = WV @ WO
print("OV.shape", VO.shape)
# fast_feats = (sae.encode_weight @ VO).flatten(0, 1)
print("sae.decode_weight.shape", sae.decode_weight.shape)
print("WO @ sae.decode_weight", (WO @ sae.decode_weight).shape)

assert t.allclose(trnsp(WV @ WO) @ sae.decode_weight, trnsp(WO) @ trnsp(WV) @ sae.decode_weight)
fast_feats = (trnsp(WV @ WO) @ sae.encode_weight.T)
print("fast_feats.shape", fast_feats.shape)
fast_feats = fast_feats.transpose(-1, -2).flatten(0, 1)
print("fast_feats.shape", fast_feats.shape)

N_FEATURES = 5
with t.no_grad():
    feature_output_logits = embeds @ fast_feats.T
print("feature_output_logits.shape", feature_output_logits.shape)
# For a random sample of features, plot the top activating tokens
for i in range(N_FEATURES):
    feat_idx = int(t.randint(0, fast_feats.shape[0], (1,)).item())
    plot_word_scores(
        feature_output_logits[:, feat_idx],
        tok_strs,
        title=f"Feature {feat_idx}: Top output logits",
        show_bottom=True,
        # non_zero_only=True
    ).show()

# %%
WQ = model.blocks[0].attn.W_Q
print("WQ.shape", WQ.shape)
WK = model.blocks[0].attn.W_K
print("WK.shape", WK.shape)

attn_scores = trnsp(sae.decode_weight) @ (WQ @ trnsp(WK)) @ sae.decode_weight

sorted_attn_scores, sorted_attn_indices = attn_scores.sort(dim=-1)
sorted_attn_indices.shape
HEAD = 0

for i in range(2):
    feat_idx = int(t.randint(0, sae.n_latents, (1,)).item())
    print(f"Feature {feat_idx}")
    plot_word_scores(
        latents[:, feat_idx],
        tok_strs,
        title=f"Feature {feat_idx}: Top activating tokens",
    ).show()
    print("Top K Attn Scores for Feature", feat_idx)
    for i in range(5):
        ith_most_attn = sorted_attn_indices[HEAD, feat_idx, i]
        plot_word_scores(
            latents[:, ith_most_attn],
            tok_strs,
            title=f"Feature {ith_most_attn}: Top activating tokens",
        ).show()