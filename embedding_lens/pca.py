#%%
from typing import Final, List, Optional, Tuple
import torch as t
from transformer_lens import HookedTransformer

from embedding_lens.embeds import get_embeds
import plotly.graph_objects as go
import plotly.express as px
import os

# from embedding_lens.visualize import plot_word_scores
from plotly.subplots import make_subplots

from embedding_lens.utils import repo_path_to_abs_path
from embedding_lens.visualize import plot_word_scores

#%%
MODEL_NAME: Final[str] = "gpt2"
# MODEL_NAME: Final[str] = "pythia-2.8b-deduped"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
#%%

def pca(X: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
    # Assume X is your data tensor of shape (n_samples, n_features)
    # Step 1: Standardize the data
    mean = t.mean(X, 0)
    std_dev = t.std(X, 0)
    X_normalized = (X - mean) / std_dev

    # Step 2: Compute the covariance matrix
    X_t = X_normalized.t()
    covariance_matrix = X_t @ X_normalized / (X_t.size(1) - 1)

    # Step 3: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = t.linalg.eigh(covariance_matrix, UPLO='U')

    # Step 4: Sort the eigenvectors by decreasing eigenvalues
    sorted_indices = t.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors


d_model = model.cfg.d_model
embeds, tok_strs = get_embeds(model, DEVICE, en_only=True)
principal_values, principal_components = pca(embeds)
print("principal_values", principal_values.shape, "principal_components", principal_components.shape)

# Plot the principal values as a bar chart
px.bar(x=t.arange(principal_values.size(0)).cpu().numpy(), y=principal_values.cpu().numpy()).show()
#%%
folder_path = repo_path_to_abs_path("figures/pca")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
N_COMPONENTS = 3
for i in range(N_COMPONENTS):
    component = principal_components[:, i]
    projected = embeds @ component
    title = f"Projecting tokens on to PCA component {i}"
    plot_word_scores(projected, tok_strs, title=title).write_image(f"{folder_path}/pca_{i}.png", scale=3)

# For N random components, project the data onto the line
N_RANDOM_COMPONENTS = 3
for i in range(N_RANDOM_COMPONENTS):
    rand_idx = int(t.randint(0, principal_components.size(1), (1,)).item())
    component = principal_components[:, rand_idx]
    projected = embeds @ component
    title = f"Projecting tokens on to PCA component {rand_idx}"
    plot_word_scores(projected, tok_strs, title=title).write_image(f"{folder_path}/pca_{rand_idx}.png", scale=3)
# %%
