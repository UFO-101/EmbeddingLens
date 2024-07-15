#%%
from typing import Final, List
from transformer_lens import HookedTransformer
import torch as t
import plotly.graph_objects as go
import plotly.express as px
from torch.nn.functional import mse_loss, cosine_similarity
import os

from embedding_lens.embeds import get_embeds
from embedding_lens.gated_sae import GatedSparseAutoencoder
from embedding_lens.lr_scheduler import get_scheduler
from embedding_lens.sae import SparseAutoencoder
from embedding_lens.custom_tqdm import tqdm
from embedding_lens.train import train
from embedding_lens.utils import repo_path_to_abs_path
from embedding_lens.visualize import plot_word_scores
#%%

MODEL_NAME: Final[str] = "gpt2"
# MODEL_NAME: Final[str] = "tiny-stories-33M"
# MODEL_NAME: Final[str] = "pythia-2.8b-deduped"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
GATED = False

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
d_model = model.cfg.d_model
embeds, tok_strs = get_embeds(model, DEVICE, en_only=True)

#%%
N_FEATURES = 2000
N_EPOCHS = 4000
L1_LAMBDA = 1e3
CORRELATION_LAMBDA = 100
LR = 1e-3

TRAIN = False
SAVE = False
LOAD = True

sae_name = "gated_sae" if GATED else "sae"
sae_name = f"{sae_name}_{MODEL_NAME}_{N_FEATURES}_feats_{N_EPOCHS}_epochs_{L1_LAMBDA}_l1_{LR}_lr_{CORRELATION_LAMBDA}_correlation"
file_name = f"{sae_name}.pth"
file_path = repo_path_to_abs_path(f"trained_saes/{file_name}")

if TRAIN:
    sae = train(embeds, N_FEATURES, d_model, LR, N_EPOCHS, L1_LAMBDA, DEVICE, GATED,
                use_correlation_loss=True, CORRELATION_LAMBDA=CORRELATION_LAMBDA)
if SAVE:
    t.save(sae.state_dict(), file_path)
if LOAD:
    if GATED:
        sae = GatedSparseAutoencoder(N_FEATURES, d_model, decoder_bias=embeds.mean(0)).to(DEVICE)
    else:
        sae = SparseAutoencoder(N_FEATURES, d_model, decoder_bias=embeds.mean(0)).to(DEVICE)
    sae.load_state_dict(t.load(file_path))

# %%
with t.no_grad():
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

folder_name = f"figures/{sae_name}/metrics"
folder_path = repo_path_to_abs_path(folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Plot distribution of reconstruction L2 distance
px.histogram(x=recons_l2_dists.cpu().numpy(), nbins=100, title="MSEs").show()
# .write_image(f"{folder_path}/gpt2_sae_mse.png", scale=3)
# Plot distribution of normalized L2 distance
px.histogram(x=normalized_l2_dists.cpu().numpy(), nbins=100, title="Normalized L2s").show()
# .write_image(f"{folder_path}/gpt2_sae_normalized_l2.png", scale=3)
# Plot distribution of cosine similarity
px.histogram(x=cosine_sims.cpu().numpy(), nbins=100, title="Cosine Similarity").show()
# .write_image(f"{folder_path}/gpt2_sae_cosine_similarity.png", scale=3)
# Plot distribution of L1s
px.histogram(x=l1s.cpu().numpy(), nbins=100, title="L1s").show()
# .write_image(f"{folder_path}/gpt2_sae_l1.png", scale=3)
# Plot distribution of non-zero latents by token
px.histogram(x=non_zero_latents.cpu().numpy(), nbins=100, title="L0s").show()
# .write_image(f"{folder_path}/gpt2_sae_l0.png", scale=3)
# Plot distribution of non_zero tokens by latent
px.histogram(
    x=non_zero_occurences.cpu().numpy(),
    nbins=100,
    title="Number of times features activated"
).show()
# ).write_image(f"{folder_path}/non_zero_occurences.png", scale=3)

#%%
N_FEATURES = 5
with t.no_grad():
    feature_output_logits = embeds @ sae.decode_weight
# For a random sample of features, plot the top activating tokens
for i in range(N_FEATURES):
    feat_idx = int(t.randint(0, sae.n_latents, (1,)).item())
    folder_name = f"figures/{sae_name}/features/feat_{feat_idx}"
    folder_path = repo_path_to_abs_path(folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plot_word_scores(
        latents[:, feat_idx],
        tok_strs,
        title=f"Feature {feat_idx}: Top activating tokens",
        # non_zero_only=True
    ).show()
    # .write_image(f"{folder_path}/top_activating_tokens.png", scale=3)
    plot_word_scores(
        feature_output_logits[:, feat_idx],
        tok_strs,
        title=f"Feature {feat_idx}: Top output logits",
        # non_zero_only=True
    ).show()
    # .write_image(f"{folder_path}/top_output_logits.png", scale=3)
# %%

N_TOKENS = 5
N_FEATURES_PER_TOKEN = 5
# For a random sample of tokens, show the top activating features
# AND for each of those features, show the top activating tokens
for i in range(N_TOKENS):
    rand_tok_idx = int(t.randint(0, embeds.shape[0], (1,)).item())
    print(f"Token {rand_tok_idx}, '{tok_strs[rand_tok_idx]}'")
    top_activations, top_activating_features = latents[rand_tok_idx].topk(N_FEATURES_PER_TOKEN)
    folder_name = f"figures/{sae_name}/tokens/{tok_strs[rand_tok_idx]}"
    folder_path = repo_path_to_abs_path(folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for rank, (feat_idx, activation) in enumerate(zip(top_activating_features, top_activations)):
        plot_word_scores(
            latents[:, feat_idx],
            tok_strs,
            title=f"Feature {feat_idx}: Top activating tokens",
        ).show()
        # .write_image(f"{folder_path}/feat_{feat_idx}_[rank_{rank}_act_{activation}]_top_activating_tokens.png", scale=3)

#%%
logits, activations = model.run_with_cache("The cat sat on the mat")
activations.keys()
acts = activations["blocks.5.hook_resid_pre"]
# acts = activations["hook_embed"]
recon_acts, intermediate = sae(acts)
recon_acts.shape

mse = mse_loss(recon_acts, acts)
cosine_sim = cosine_similarity(recon_acts, acts, dim=-1)
l0 = (intermediate != 0).sum(dim=-1)
print("mse", mse.tolist())
print("cosine_sim", cosine_sim.tolist())
print("l0", l0.tolist())

#%%