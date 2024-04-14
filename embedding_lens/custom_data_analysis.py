#%%
from typing import Final, List
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
from embedding_lens.embeds import get_embeds
from embedding_lens.sae import SparseAutoencoder
from embedding_lens.utils import repo_path_to_abs_path
from transformer_lens import HookedTransformer
import torch as t
import pandas as pd
#%%
spell = SpellChecker()

MODEL_NAME: Final[str] = "gpt2"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
d_model = model.cfg.d_model
embeds, tok_strs = get_embeds(model, DEVICE, en_only=True)
N_FEATURES = 2000
N_EPOCHS = 25000
L1_LAMBDA = 2e-1
LR = 1e-3

file_name = f"sae_{MODEL_NAME}_{N_FEATURES}_feats_{N_EPOCHS}_epochs_{L1_LAMBDA}_l1_{LR}_lr.pth"
file_path = repo_path_to_abs_path(f"trained_saes/{file_name}")

sae = SparseAutoencoder(N_FEATURES, d_model, decoder_bias=embeds.mean(0)).to(DEVICE)
sae.load_state_dict(t.load(file_path, map_location = DEVICE))

with t.no_grad():
    recons, l1s, latents = sae(embeds)

top_activations, top_activating_tokens = latents.topk(10)
# %%
top_activations.shape
# %%
top_activating_tokens.shape
# %%
latents.shape
# %%
top_activating_tokens

# %%
df_contigency = pd.DataFrame(t.nonzero(latents).numpy())
df_contigency.columns = ['token_index', 'feature_index']
# %%
df_contigency.head()
len(df_contigency)
# %%
def plot_distribution(df_contigency, group_by_col, count_of_col, num_bins = 10, graph_title = None):
    # turn contingency matrix dataframe into value_counts that are ready to be plotted
    token_to_feature_counts= pd.DataFrame(df_contigency.groupby(f'{group_by_col}_index').count())
    token_to_feature_counts.reset_index(inplace=True)
    token_to_feature_counts.columns = [f'{group_by_col}_index', f'num_of_{count_of_col}']
    fired_features_bins = pd.cut(token_to_feature_counts[f'num_of_{count_of_col}'], num_bins)
    value_counts = fired_features_bins.value_counts()
    value_counts.sort_index(inplace = True) 
    # plot historgram
    fig, ax = plt.subplots(figsize=(20, 6)) 
    value_counts.plot(kind='bar', color='skyblue')
    if graph_title is None:
        plt.title(f'Histogram of number of {count_of_col} per {group_by_col}')
    else:
        plt.title(graph_title)
    plt.xlabel(f'Num of {count_of_col} for each {group_by_col}')
    plt.ylabel(f'Count of {group_by_col}s')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.show()
    return
# %%
plot_distribution(df_contigency,
                   group_by_col = 'token',
                   count_of_col = 'activated_features')
# %%
plot_distribution(df_contigency,
                   group_by_col = 'feature',
                   count_of_col = 'associated_tokens')

# # %%
# top_activating_tokens
# # %%
# # Look at co-activating patterns between tokens
# # df_contigency.head()
# # df_contigency.groupby(['feature_index'])
# # Create a binary matrix where rows represent tokens and columns represent features
# sets_list = [set(indices.tolist()) for indices in top_activating_tokens[:1000]]
# sets_list
# # %%
# # Initialize matrix to store overlap counts
# overlap_counts = t.zeros(len(sets_list), len(sets_list))

# # Iterate over pairs of sets and count overlapping elements
# for i in range(len(sets_list)):
#     for j in range(i+1, len(sets_list)):
#         overlap_counts[i, j] = len(sets_list[i].intersection(sets_list[j]))
#         overlap_counts[j, i] = overlap_counts[i, j]  # Symmetric matrix

# print(overlap_counts)
# # %%
# overlap_counts.shape
# # # # %%
# # import networkx as nx
# # from networkx.algorithms import bipartite
# # from pyvis.network import Network
# # # %%
# # # Create a Network instance
# # B = nx.Graph()
# # B.add_nodes_from(df_contigency['token_index'].values.tolist(), bipartite = 0)
# # B.add_nodes_from(df_contigency['feature_index'].values.tolist(), bipartite = 1)

# # B.add_edges_from(list(zip(df_contigency['token_index'], df_contigency['feature_index'])))

# # # %%
# # B.edges(20)
# # B.nodes(20)
# # nx.is_bipartite(B)
# # %%
# # nt = Network('500px', '500px')
# # nt.from_nx(B)
# # nt.show('../graphs/nx.html',notebook=False)
# # # %%
# # top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
# # bottom_nodes = set(B) - top_nodes
# # # # %%


# # %%
