#%%
from typing import Final, List
from transformer_lens import HookedTransformer
import torch as t
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import plotly.express as px
from torch.nn.functional import mse_loss, cosine_similarity
from spellchecker import SpellChecker

from embedding_lens.lr_scheduler import get_scheduler
from embedding_lens.sae import SparseAutoencoder
from embedding_lens.custom_tqdm import tqdm
from embedding_lens.utils import repo_path_to_abs_path
#%%

spell = SpellChecker()
MODEL_NAME: Final[str] = "gpt2"
# MODEL_NAME: Final[str] = "pythia-2.8b-deduped"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
d_model = model.cfg.d_model
embeds = model.W_E.detach().clone().to(DEVICE)

#%%
# filter to english words
tok_strs: List[str] = model.to_str_tokens(t.arange(model.cfg.d_vocab))  # type: ignore
tok_strs = [word.strip() for word in tok_strs]
correct_words = spell.known(tok_strs)
en_idxs = [i for i, tok_str in enumerate(tok_strs) if tok_str.strip() in correct_words]
print("embeds", embeds.shape)
embeds = embeds[en_idxs]
print("en embeds", embeds.shape)

# embeds_dataset = TensorDataset(embeds)
# embeds_dataloader = DataLoader(embeds_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

#%%
# BATCH_SIZE = 512
N_FEATURES = 2500 # first is 10,000, then 5000
N_EPOCHS = 25000
L1_LAMBDA = 2e-1
LR = 1e-3

TRAIN = False
SAVE = False
LOAD = True
file_name = f"sae_{MODEL_NAME}_{N_FEATURES}_feats_{N_EPOCHS}_epochs_{L1_LAMBDA}_l1_{LR}_lr.pth"
file_path = repo_path_to_abs_path(f"trained_saes/{file_name}")

if TRAIN:
    sae = SparseAutoencoder(N_FEATURES, d_model, decoder_bias=embeds.mean(0)).to(DEVICE)
    optim = t.optim.Adam(sae.parameters(), lr=LR)
    scheduler = get_scheduler(
        scheduler_name="CosineAnnealingWarmRestarts",
        optimizer=optim,
        # training_steps=N_EPOCHS * len(embeds_dataloader),
        training_steps=N_EPOCHS,
        lr = 4e-4,
        warm_up_steps=0,
        decay_steps=0,
        lr_end=0.0,
        num_cycles=50,
    )

    step_history, mse_history, l1_history, loss_history, lr_history = [], [], [], [], []
    n_dead_feature_history = []

    step=0
    for epoch in tqdm(range(N_EPOCHS)):
        # for batch in tqdm(embeds_dataloader):
            # embed_batch = batch[0]
            embed_batch = embeds
            reconstruction, l1s, latents = sae(embed_batch)
            mses = (reconstruction - embed_batch).pow(2).sum(dim=-1)
            mse_term = mses.mean()
            l1_term = L1_LAMBDA * l1s.mean()
            loss = mse_term + l1_term
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Make the decoder weights have column-wise unit norm
            sae.decode_weight.data /= sae.decode_weight.data.norm(dim=0)
            step += 1
            scheduler.step()

            dead_features = latents.sum(dim=0) == 0
            n_dead_features = dead_features.sum().item()
            if step % 1 == 0:
                step_history.append(step)
                mse_history.append(mse_term.item())
                l1_history.append(l1s.mean().item())
                loss_history.append(loss.item())
                lr_history.append(optim.param_groups[0]["lr"])
                n_dead_feature_history.append(n_dead_features)

            top_k_mses, top_k_indices = mses.topk(n_dead_features)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_history, y=loss_history, name="Loss"))
    fig.add_trace(go.Scatter(x=step_history, y=mse_history, name="MSE"))
    fig.add_trace(go.Scatter(x=step_history, y=l1_history, name="L1"))
    fig.add_trace(go.Scatter(x=step_history, y=lr_history, name="LR"))
    fig.add_trace(go.Scatter(x=step_history, y=n_dead_feature_history, name="Dead Features"))
    fig.show()

if SAVE:
    t.save(sae.state_dict(), file_path)
if LOAD:
    sae = SparseAutoencoder(N_FEATURES, d_model, decoder_bias=embeds.mean(0)).to(DEVICE)
    sae.load_state_dict(t.load(file_path))

# %%
with t.no_grad():
    recons, l1s, latents = sae(embeds)
    print("recons", recons.shape, "latents", latents.shape)
    recons_l2_dists = (recons - embeds).pow(2).sum(dim=-1)
    print("recons_l2_dist", recons_l2_dists.shape)
    cosine_sims = cosine_similarity(recons, embeds, dim=-1)
    print("cosine_sims", cosine_sims.shape)
    non_zero_latents = (latents != 0).sum(dim=-1)
    print("non_zero_latents", non_zero_latents.shape)
    non_zero_occurences = (latents != 0).sum(dim=0)
    print("non_zero_occurences", non_zero_occurences.shape)

# Plot distribution of reconstruction L2 distance
px.histogram(x=recons_l2_dists.cpu().numpy(), nbins=100, title="MSEs").show()
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
N_FEATURES = 25
# For a random sample of features, plot the top activating tokens
for i in range(N_FEATURES):
    feat_idx = t.randint(0, sae.n_latents, (1,)).item()
    print(f"Feature {feat_idx}", "n activated", non_zero_occurences[feat_idx].item())
    top_activations, top_activating_tokens = latents[:, feat_idx].topk(10)
    top_activating_tokens = t.tensor([en_idxs[i] for i in top_activating_tokens])
    feat_tok_strs = model.to_str_tokens(top_activating_tokens)
    for tok_str, activation in zip(feat_tok_strs, top_activations):
        print(f"'{tok_str}' {activation.item():.2f}")
    print()