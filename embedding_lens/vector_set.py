#%%
import os
from typing import Final
from embedding_lens.binary_autoencoder import BinaryAutoencoder
from transformer_lens import HookedTransformer
from embedding_lens.embeds import get_embeds
from embedding_lens.lr_scheduler import get_scheduler
import torch as t
from custom_tqdm import tqdm
import plotly.graph_objects as go
from einops import einsum
from torch.nn.functional import mse_loss, cosine_similarity
import plotly.express as px

from embedding_lens.toy_datasets import CombinationDataset, FeatureSet
from embedding_lens.utils import repo_path_to_abs_path
from embedding_lens.visualize import plot_word_scores
from torch.utils.data import DataLoader

MODEL_NAME: Final[str] = "gpt2"
# MODEL_NAME: Final[str] = "tiny-stories-33M"
# MODEL_NAME: Final[str] = "pythia-2.8b-deduped"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
GATED = False

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
# d_model = model.cfg.d_model
d_model = 2


def two_d_vector_plot(vectors, title=""):
    """Vectors should have shape (n_vectors, 2)"""
    fig = go.Figure()
    # Add each vector
    for i in range(vectors.size(0)):
        fig.add_trace(
            go.Scatter(x=[0, vectors[i, 0].item()],
                        y=[0, vectors[i, 1].item()],
                        mode='lines+markers',
                        name=f'Vector {i+1}'
        ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # Add the title
    fig.update_layout(title=title)
    fig.show()



dataset = CombinationDataset(
    [
        # FeatureSet.from_default(4, sparsity=0.5),
        FeatureSet.within_range(1, feature_range=(1, 1), sparsity=0.5),
        FeatureSet.within_range(1, feature_range=(1, 1), sparsity=0.5),
    ],
    size=1000,
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
embeds = t.stack([dataset[i] for i in range(1000)]).to(DEVICE)

# embeds, tok_strs = get_embeds(model, DEVICE, en_only=True)

#%%

def train(
    embeds: t.Tensor,
    N_FEATURES: int,
    d_model: int,
    LR: float,
    N_EPOCHS: int,
    DEVICE: str,
    lr_schedule: bool = False,
) -> t.Tensor:
    directions = t.nn.Parameter(t.randn(N_FEATURES, d_model).to(DEVICE))
    optim = t.optim.Adam([directions], lr=LR)
    scheduler = get_scheduler(
        scheduler_name="CosineAnnealingWarmRestarts",
        optimizer=optim,
        # training_steps=N_EPOCHS * len(embeds_dataloader),
        training_steps=N_EPOCHS,
        lr = 4e-2,
        warm_up_steps=0,
        decay_steps=N_EPOCHS // 5,
        lr_end=LR / 100,
        num_cycles=50,
    )

    step_history, mse_history, mask_history, loss_history, lr_history = [], [], [], [], []
    n_dead_feature_history = []

    step=0
    for epoch in (pbar:=tqdm(range(N_EPOCHS))):
        embed_batch = embeds
        cosines = t.nn.functional.cosine_similarity(embed_batch[None], directions[:, None], dim=-1)
        max_cosines, _ = cosines.max(dim=-1)
        loss = -(((1 + max_cosines)/ 2).pow(5)).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"loss {loss.item():.2f}")

        # bae.encode_weight.data /= bae.encode_weight.data.norm(dim=-1, keepdim=True)
        directions.data /= directions.data.norm(dim=-1, keepdim=True)

        # Make the decoder weights have column-wise unit norm
        step += 1
        if lr_schedule:
            scheduler.step()

        if step % 10 == 0:
            step_history.append(step)
            loss_history.append(loss.item())
            lr_history.append(optim.param_groups[0]["lr"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_history, y=loss_history, name="Loss"))
    fig.add_trace(go.Scatter(x=step_history, y=mse_history, name="MSE"))
    fig.add_trace(go.Scatter(x=step_history, y=mask_history, name="Mask"))
    fig.add_trace(go.Scatter(x=step_history, y=lr_history, name="LR"))
    fig.add_trace(go.Scatter(x=step_history, y=n_dead_feature_history, name="Dead Features"))
    fig.show()
    return directions


N_FEATURES = 10
N_EPOCHS = 10000
MASK_LAMBDA = 1e-1
# MASK_LAMBDA = 0.3
# LR = 2e-1
LR = 0.2

TRAIN = True
SAVE = False
LOAD = False

directions_name = f"bae_{MODEL_NAME}_{N_FEATURES}_feats_{N_EPOCHS}_epochs_{LR}_lr"
file_name = f"{directions_name}.pth"
file_path = repo_path_to_abs_path(f"trained_saes/{file_name}")

if TRAIN:
    directions = train(embeds, N_FEATURES, d_model, LR, N_EPOCHS, DEVICE, True)
    
    # Claude's analytical solution
    # m = (embeds @ embeds.T) / t.trace(embeds @ embeds.T)
    # print("embeds", embeds.shape, "m", m.shape)
    # eigvals, eigvecs = t.linalg.eig(m)
    # print("eigvals", eigvals.shape, "eigvecs", eigvecs.shape)
    # directions = eigvecs[:, :N_FEATURES]
    # print("directions", directions.shape)

    # ChatGPT's analytical solution
    # U, S, V_T = t.svd(embeds)
    # print("embeds", embeds.shape, "U", U.shape, "S", S.shape, "V_T", V_T.shape)
    # directions = V_T[:N_FEATURES]
    # print("directions", directions.shape)
if SAVE:
    t.save(directions, file_path)
if LOAD:
    directions = t.load(file_path)

two_d_vector_plot(directions, title="Encode weights")

# %%
with t.no_grad():
    recons, latents, latent_pre_acts, latent_mask = bae(embeds)
    two_d_vector_plot(bae.encode_weight, title="Encode weights")
    two_d_vector_plot((bae.decode_weight * bae.feature_mags).T, title="Decode Weights")
    two_d_vector_plot(embeds[:10], title="Input data")
    two_d_vector_plot(recons[:10], title="BAE Reconstructed data")
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
    feature_output_logits = embeds @ bae.decode_weight
# For a random sample of features, plot the top activating tokens
for i in range(N_FEATURES):
    feat_idx = int(t.randint(0, bae.n_latents, (1,)).item())
    folder_name = f"figures/{bae_name}/features/feat_{feat_idx}"
    folder_path = repo_path_to_abs_path(folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plot_word_scores(
        latent_pre_acts[:, feat_idx],
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
