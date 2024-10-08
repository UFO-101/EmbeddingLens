# %%
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

from embedding_lens import top_activating_examples
from embedding_lens.custom_tqdm import tqdm
from embedding_lens.embeds import get_embeds
from embedding_lens.fast_features import attn_0_derived_features
from embedding_lens.linear_model_sae import no_layernorm_gpt2
from embedding_lens.lr_scheduler import get_scheduler
from embedding_lens.reverse_top_k_sae import ReverseTopKSparseAutoencoder
from embedding_lens.top_activating_examples import get_top_activating_examples
from embedding_lens.utils import repo_path_to_abs_path
from embedding_lens.visualize import plot_direction_unembeds, plot_word_scores
from tuned_lens import TunedLens

from transformers import AutoModelForCausalLM
# %%

MODEL_NAME: Final[str] = "gpt2"
# MODEL_NAME: Final[str] = "gemma-2-2b"

# MODEL_NAME: Final[str] = "tiny-stories-33M"
DEVICE = "cuda:3" if t.cuda.is_available() else "cpu"
# DEVICE = "cpu"
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
huggingface_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tuned_lens = TunedLens.from_model_and_pretrained(huggingface_model).to(DEVICE)
del huggingface_model
# %%
d_model = model.cfg.d_model
EN_ONLY = False
embeds, tok_strs = get_embeds(model, DEVICE, en_only=EN_ONLY)
tok_strs: List[str] = model.to_str_tokens(t.arange(model.cfg.d_vocab).int())  # type: ignore
# embeds = attn_0_derived_features(model, embeds.T)
MLP_IDX = 3
# embeds = model.blocks[MLP_IDX].mlp.W_in.T


# %%

def train(
    embeds: t.Tensor,
    N_FEATURES: int,
    d_model: int,
    LR: float,
    N_EPOCHS: int,
    batch_size: int,
    DEVICE: str,
    lr_schedule: bool = False,
    zipf_coeffs: Tuple[float, float] = (1.0, 2.7),
) -> ReverseTopKSparseAutoencoder:
    # assert embeds.shape[0] > N_FEATURES
    sae = ReverseTopKSparseAutoencoder(
        n_latents=N_FEATURES,
        n_inputs=d_model,
        batch_size=batch_size,
        zipf_coeffs=zipf_coeffs,
        decoder_bias=embeds.mean(0),
    ).to(DEVICE)
    optim = t.optim.Adam(sae.parameters(), lr=LR)
    scheduler = get_scheduler(
        scheduler_name="CosineAnnealingWarmRestarts",
        optimizer=optim,
        training_steps=N_EPOCHS,
        lr=4e-2,
        warm_up_steps=0,
        decay_steps=0,
        lr_end=0.0,
        num_cycles=min(20, N_EPOCHS // 10),
    )
    dataset = TensorDataset(embeds.detach())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    step_history, mse_history, loss_history, lr_history = [], [], [], []
    n_dead_feature_history = []

    step = 0
    for epoch in (pbar := tqdm(range(N_EPOCHS))):
        for embed_batch in loader:
            embed_batch = embed_batch[0]
            if step == 0:
                print("embed_batch", embed_batch.shape)
            reconstruction, intermediate = sae(embed_batch)
            mses = (reconstruction - embed_batch).pow(2).sum(dim=-1)
            mse_term = mses.mean()
            loss = mse_term

            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_description(f"mse {mse_term.item():.2f} loss {loss.item():.2f}")

            dead_features = intermediate.sum(dim=0) == 0
            n_dead_features = dead_features.sum().item()
            if step % 10 == 0:
                step_history.append(step)
                mse_history.append(mse_term.item())
                loss_history.append(loss.item())
                lr_history.append(optim.param_groups[0]["lr"])
                n_dead_feature_history.append(n_dead_features)

            step += 1
            if lr_schedule:
                scheduler.step()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_history, y=loss_history, name="Loss"))
    fig.add_trace(go.Scatter(x=step_history, y=mse_history, name="MSE"))
    fig.add_trace(go.Scatter(x=step_history, y=lr_history, name="LR"))
    fig.add_trace(
        go.Scatter(x=step_history, y=n_dead_feature_history, name="Dead Features")
    )
    fig.show()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig.write_image(
        repo_path_to_abs_path(
            f"trained_saes/{name}_loss_mse_lr_dead_features_{timestamp}.png"
        )
    )
    return sae


# WHY ARE MSEs SO HIGH?

TRAIN = False
SAVE = False
LOAD = True

N_FEATURES = 2000
N_EPOCHS = 3_000
# N_EPOCHS = 20
# BATCH_SIZE = 30_000
BATCH_SIZE = embeds.shape[0]
LR = 4e-2
# ZIPF_COEFFS = (1.0, 100.0)
# ZIPF_COEFFS = (1.0, 7.0 * 4.5)
# ZIPF_COEFFS = (1.0, 7.0 * 7)
ZIPF_COEFFS = (1.0, 7.0)


sae_name = f"reverse_topk_zipf_{ZIPF_COEFFS[0]}_a_{ZIPF_COEFFS[1]}_b"
# name = f"{sae_name}_{MODEL_NAME}_{N_FEATURES}_en_only_{EN_ONLY}_feats_{N_EPOCHS}_epochs_{LR}_lr"
name = f"{sae_name}_{MODEL_NAME}_{N_FEATURES}_final_MLP_Wout_feats_{N_EPOCHS}_epochs_{LR}_lr"
file_path = repo_path_to_abs_path(f"trained_saes/{name}.pth")

if TRAIN:
    sae = train(
        embeds=embeds,
        N_FEATURES=N_FEATURES,
        d_model=d_model,
        LR=LR,
        N_EPOCHS=N_EPOCHS,
        batch_size=BATCH_SIZE,
        DEVICE=DEVICE,
        lr_schedule=False,
        zipf_coeffs=ZIPF_COEFFS,
    )
if SAVE:
    t.save(sae.state_dict(), file_path)
if LOAD:
    file_path = repo_path_to_abs_path(
        # f"trained_saes/sae_gpt2_2000_feats_25000_epochs_0.2_l1_0.001_lr.pth"
        # "trained_saes/reverse_topk_zipf_1.0_a_31.5_b_gemma-2-2b_4000_en_only_False_feats_10000_epochs_0.04_lr.pth"
        # "trained_saes/reverse_topk_zipf_1.0_a_31.5_b_gpt2_300_final_MLP_Wout_feats_10000_epochs_0.04_lr.pth"
        # "trained_saes/reverse_topk_zipf_1.0_a_7.0_b_1000_feats_100000_epochs_0.01_lr.pth"
        "trained_saes/reverse_topk_zipf_1.0_a_7.0_b_gpt2_2000_en_only_False_feats_3000_epochs_0.04_lr.pth"
    )
    sae = ReverseTopKSparseAutoencoder(N_FEATURES, d_model, BATCH_SIZE, ZIPF_COEFFS).to(
        DEVICE
    )
    sae.load_state_dict(t.load(file_path, map_location=DEVICE))
    # sae = ReverseTopKSparseAutoencoder(20518, d_model, embeds.shape[0], (1.0, 21000))
    # sae.to(DEVICE)
    # state_dict = {
    #     "encode_weight": embeds,
    #     "decode_weight": embeds.T,
    #     "dec_bias": t.zeros_like(embeds.mean(0), device=DEVICE),
    # }
    # sae.load_state_dict(state_dict)
#%%
fast_features = attn_0_derived_features(model, sae.decode_weight)
get_top_activating_examples(model, fast_features, "blocks.0.hook_resid_mid", batch_size=10)

# %%
with t.no_grad():
    mean_embed = embeds.mean(0)
    dataset = TensorDataset(embeds.detach())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print("hi")
    input_embeds = next(iter(loader))[0]
    recons, latents = sae(input_embeds)
    l1s = t.abs(latents).sum(dim=-1)
    print("recons", recons.shape, "latents", latents.shape)
    recons_l2_dists = (recons - input_embeds).pow(2).sum(dim=-1)
    input_embeds_var = (input_embeds - mean_embed).pow(2).sum(dim=-1).mean()
    print("recons_l2_dist", recons_l2_dists.shape)
    normalized_l2_dists = recons_l2_dists / input_embeds_var
    print("normalized_l2_dists", normalized_l2_dists.shape)
    cosine_sims = cosine_similarity(recons, input_embeds, dim=-1)
    print("cosine_sims", cosine_sims.shape)
    non_zero_latents = (latents != 0).sum(dim=-1)
    print("non_zero_latents", non_zero_latents.shape)
    non_zero_occurences = (latents != 0).sum(dim=0)
    print("non_zero_occurences", non_zero_occurences.shape)

# Plot distribution of reconstruction L2 distance
fig = px.histogram(x=recons_l2_dists.cpu().numpy(), nbins=100, title="MSEs")
fig.add_vline(
    x=(mean := recons_l2_dists.mean().item()), line_dash="dash", line_color="red"
)
fig.add_annotation(
    x=mean,
    y=0.15,
    yref="paper",
    text=f"Mean: {mean:.2f}",
    showarrow=False,
    font=dict(color="white"),
)
fig.show()

# Plot distribution of normalized L2 distance
fig = px.histogram(
    x=normalized_l2_dists.cpu().numpy(),
    nbins=100,
    title="Fraction of Variance Unexplained (FVU)",
)
fig.add_vline(
    x=(mean := normalized_l2_dists.mean().item()), line_dash="dash", line_color="red"
)
fig.add_annotation(
    x=mean,
    y=0.15,
    yref="paper",
    text=f"Mean: {mean:.2f}",
    showarrow=False,
    font=dict(color="white"),
)
fig.show()
# Plot distribution of cosine similarity
fig = px.histogram(x=cosine_sims.cpu().numpy(), nbins=100, title="Cosine Similarity")
fig.add_vline(x=(mean := cosine_sims.mean().item()), line_dash="dash", line_color="red")
fig.add_annotation(
    x=mean,
    y=0.15,
    yref="paper",
    text=f"Mean: {mean:.2f}",
    showarrow=False,
    font=dict(color="white"),
)
fig.show()
# Plot distribution of L1s
fig = px.histogram(x=l1s.cpu().numpy(), nbins=100, title="L1s")
fig.add_vline(x=(mean := l1s.mean().item()), line_dash="dash", line_color="red")
fig.add_annotation(
    x=mean,
    y=0.15,
    yref="paper",
    text=f"Mean: {mean:.2f}",
    showarrow=False,
    font=dict(color="white"),
)
fig.show()
# Plot distribution of non-zero latents by token
fig = px.histogram(x=non_zero_latents.cpu().numpy(), nbins=100, title="L0s")
fig.add_vline(
    x=(mean := non_zero_latents.float().mean().item()),
    line_dash="dash",
    line_color="red",
)
fig.add_annotation(
    x=mean,
    y=0.15,
    yref="paper",
    text=f"Mean: {mean:.2f}",
    showarrow=False,
    font=dict(color="black"),
)
fig.show()
# Plot distribution of non_zero tokens by latent
fig = px.histogram(
    x=non_zero_occurences.cpu().numpy(),
    nbins=100,
    title="Number of times features activated",
)
fig.add_vline(
    x=(mean := non_zero_occurences.float().mean().item()),
    line_dash="dash",
    line_color="red",
)
fig.add_annotation(
    x=mean,
    y=0.15,
    yref="paper",
    text=f"Mean: {mean:.2f}",
    showarrow=False,
    font=dict(color="black"),
)
fig.show()

# %%
N_FEATURES = 5
with t.no_grad():
    tuned_output_logits = tuned_lens(sae.decode_weight.T, MLP_IDX + 1).T
    feature_output_logits = model.unembed.W_U.T @ sae.decode_weight
# For a random sample of features, plot the top activating tokens
# for i in range(N_FEATURES):
for i in [0, 3, 13, 23, 54, 103]:
    feat_idx = int(t.randint(0, sae.n_latents, (1,)).item())
    # plot_word_scores(
    #     latents[:, feat_idx],
    #     tok_strs,
    #     title=f"Feature {feat_idx}: Top activating tokens",
    #     # show_bottom=True,
    # ).show()
    plot_word_scores(
        feature_output_logits[:, feat_idx],
        tok_strs,
        title=f"Feature {feat_idx}: Top output logits",
        show_bottom=True,
    ).show()
    plot_word_scores(
        tuned_output_logits[:, feat_idx],
        tok_strs,
        title=f"Tuned {feat_idx}: Top output logits",
        show_bottom=True,
    ).show()
    print("/n/n")
# %%

N_TOKENS = 5
N_FEATURES_PER_TOKEN = 5
# For a random sample of tokens, show the top activating features
# AND for each of those features, show the top activating tokens
for i in range(N_TOKENS):
    rand_tok_idx = int(t.randint(0, embeds.shape[0], (1,)).item())
    print(f"Token {rand_tok_idx}, '{tok_strs[rand_tok_idx]}'")
    top_activations, top_activating_features = latents[rand_tok_idx].topk(
        N_FEATURES_PER_TOKEN
    )
    for rank, (feat_idx, activation) in enumerate(
        zip(top_activating_features, top_activations)
    ):
        plot_word_scores(
            latents[:, feat_idx],
            tok_strs,
            title=f"Feature {feat_idx}: Top activating tokens",
            show_bottom=True,
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

assert t.allclose(
    trnsp(WV @ WO) @ sae.decode_weight, trnsp(WO) @ trnsp(WV) @ sae.decode_weight
)
fast_feats = trnsp(WV @ WO) @ sae.encode_weight.T
print("fast_feats.shape", fast_feats.shape)
fast_feats = fast_feats.transpose(-1, -2)
print("fast_feats.shape", fast_feats.shape)

N_FEATURES = 5
HEAD = 2
with t.no_grad():
    ov_feature_output_logits = einsum(embeds, fast_feats, "t d, h f d -> t h f")
print("ov_feature_output_logits.shape", ov_feature_output_logits.shape)
# For a random sample of features, plot the top activating tokens
for i in range(N_FEATURES):
    feat_idx = int(t.randint(0, ov_feature_output_logits.shape[-1], (1,)).item())
    print(f"Feature {feat_idx}")
    plot_word_scores(
        feature_output_logits[:, feat_idx],
        tok_strs,
        title=f"Feature {feat_idx}: Top output logits",
    ).show()
    print(f"Head {HEAD} Feature {feat_idx}")
    plot_word_scores(
        ov_feature_output_logits[:, HEAD, feat_idx],
        tok_strs,
        title=f"Head {HEAD} Feature {feat_idx}: Top output logits",
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
