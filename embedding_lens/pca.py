#%%
from typing import Final, List, Optional, Tuple
import torch as t
from transformer_lens import HookedTransformer

from embedding_lens.embeds import get_embeds
import plotly.graph_objects as go
import plotly.express as px

# from embedding_lens.visualize import plot_word_scores
from plotly.subplots import make_subplots

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
def plot_word_scores(
    scores: t.Tensor,
    words: List[str],
    list_len: int = 10,
    n_cols: int = 3,
    title: Optional[str] = None
) -> go.Figure:
    assert scores.size(0) == len(words), "Scores and words must have the same length"
    fig = make_subplots(rows=2, cols=n_cols,
                        specs=[[{"colspan": 3}, None, None],
                               [{}, {}, {}]],
                        row_heights=[0.2, 0.8],
    )
    # Histogram of scores in the top row
    fig.add_trace(
        go.Histogram(x=scores.cpu().numpy(), nbinsx=100),
        row=1, col=1
    )
    fig.update_yaxes(tickvals=[], ticktext=[], row=1, col=1)

    # Get the top, middle, and bottom `list_len` words
    top_scores, top_idxs = scores.topk(list_len)

    # Sort the scores and get indices
    sorted_scores, sorted_indices = scores.sort()
    # Calculate the start and end indices for the middle scores
    middle_start = len(scores) // 2 - list_len // 2
    middle_end = middle_start + list_len
    # Extract the middle scores and words
    middle_scores = sorted_scores[middle_start:middle_end]
    middle_idxs = sorted_indices[middle_start:middle_end]

    bottom_scores, bottom_idxs = scores.topk(list_len, largest=False)
    min_score, max_score = scores.min().item(), scores.max().item()
    

    # In the bottom row plot the top, middle, and bottom words as heatmaps
    for i, (s, w) in enumerate(zip([top_scores, middle_scores, bottom_scores],
                                            [top_idxs, middle_idxs, bottom_idxs])):
        # Sort s and w by s
        sorted_s, sorted_idxs = s.sort()
        sorted_w = w[sorted_idxs]
        fig.add_trace(
            go.Heatmap(
                z=sorted_s.unsqueeze(-1).cpu().numpy(),
                text=[[words[i]] for i in sorted_w.cpu().numpy()],
                texttemplate="%{text}",
                textfont={"size": 15},
                zmin=min_score,
                zmax=max_score,
            ),
            row=2, col=n_cols - i
        )
        fig.update_xaxes(tickvals=[], ticktext=[], row=2, col=n_cols - i)
        fig.update_yaxes(tickvals=[], ticktext=[], row=2, col=n_cols - i)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(height=400, width=450)
    return fig

N_COMPONENTS = 1
# For N random components, project the data onto the line
for i in range(N_COMPONENTS):
    component = principal_components[:, i]
    projected = embeds @ component
    plot_word_scores(projected, tok_strs).show()
    # fig = px.scatter(x=projected.cpu().numpy(), y=t.zeros_like(projected).cpu().numpy(), hover_name=tok_strs)
    # fig.update_traces(textposition='top center')
    # fig.update_layout(title=f"Principal Component {i}")
    # fig.show()