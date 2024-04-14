from embedding_lens.lr_scheduler import get_scheduler
from embedding_lens.sae import SparseAutoencoder
import torch as t
from custom_tqdm import tqdm
import plotly.graph_objects as go


def train(
    embeds: t.Tensor,
    N_FEATURES: int,
    d_model: int,
    LR: float,
    N_EPOCHS: int,
    L1_LAMBDA: float,
    DEVICE: str
) -> SparseAutoencoder:
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

        # top_k_mses, top_k_indices = mses.topk(n_dead_features)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_history, y=loss_history, name="Loss"))
    fig.add_trace(go.Scatter(x=step_history, y=mse_history, name="MSE"))
    fig.add_trace(go.Scatter(x=step_history, y=l1_history, name="L1"))
    fig.add_trace(go.Scatter(x=step_history, y=lr_history, name="LR"))
    fig.add_trace(go.Scatter(x=step_history, y=n_dead_feature_history, name="Dead Features"))
    fig.show()
    return sae