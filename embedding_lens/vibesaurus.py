from typing import Final

from spellchecker import SpellChecker
from embedding_lens.sae import SparseAutoencoder
from embedding_lens.utils import repo_path_to_abs_path
from transformer_lens import HookedTransformer
import torch as t
import nltk
from nltk.stem import PorterStemmer
import json
import plotly.graph_objects as go

spell = SpellChecker()

MODEL_NAME: Final[str] = "gpt2"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
d_model = model.cfg.d_model
embeds = model.W_E.detach().clone().to(DEVICE)
N_FEATURES = 2000
N_EPOCHS = 25000
L1_LAMBDA = 2e-1
LR = 1e-3

# filter to english words
tok_strs = model.to_str_tokens(t.arange(model.cfg.d_vocab))  # type: ignore
tok_strs = [word.strip() for word in tok_strs]
correct_words = spell.known(tok_strs)
en_idxs = [i for i, tok_str in enumerate(tok_strs) if tok_str.strip() in correct_words]
print("embeds", embeds.shape)
embeds = embeds[en_idxs]
print("en embeds", embeds.shape)

file_name = f"sae_{MODEL_NAME}_{N_FEATURES}_feats_{N_EPOCHS}_epochs_{L1_LAMBDA}_l1_{LR}_lr.pth"
file_path = repo_path_to_abs_path(f"trained_saes/{file_name}")

sae = SparseAutoencoder(N_FEATURES, d_model, decoder_bias=embeds.mean(0)).to(DEVICE)
sae.load_state_dict(t.load(file_path, map_location=t.device('cpu') ))

with t.no_grad():
    recons, l1s, latents = sae(embeds)
    print("recons", recons.shape, "latents", latents.shape)
    recons_l2_dists = (recons - embeds).pow(2).sum(dim=-1)
    print("recons_l2_dist", recons_l2_dists.shape)
    non_zero_latents = (latents != 0).sum(dim=-1)
    print("non_zero_latents", non_zero_latents.shape)
    non_zero_occurences = (latents != 0).sum(dim=0)
    print("non_zero_occurences", non_zero_occurences.shape)

#%%

nltk.download("punkt")
ps = PorterStemmer()

unique_stems = 0
unique_stems_dict = {}
stem_set_dict = {}
for i in range(N_FEATURES):
    feat_idx = t.randint(0, sae.n_latents, (1,)).item()
    top_activations, top_activating_tokens = latents[:, feat_idx].topk(10)
    top_activating_tokens = t.tensor([en_idxs[i] for i in top_activating_tokens])
    feat_tok_strs = model.to_str_tokens(top_activating_tokens)
    stem_set = set()
    for tok_str, activation in zip(feat_tok_strs, top_activations): # arr of tuples
        stem_set.add(ps.stem(tok_str.strip()))
    if len(stem_set) == 10:
        unique_stems += 1
        #print(f"Feature {feat_idx}", "n activated", non_zero_occurences[feat_idx].item())
        unique_stems_dict[f"feat-{feat_idx}-nact{non_zero_occurences[feat_idx].item()}"] = []
        for tok_str, activation in zip(feat_tok_strs, top_activations): # arr of tuples
            #ßprint(f"'{tok_str}' {activation.item():.2f}")
            unique_stems_dict[f"feat-{feat_idx}-nact{non_zero_occurences[feat_idx].item()}"].append((tok_str, activation.item()))
        #print()
    if len(stem_set) not in stem_set_dict:
        stem_set_dict[len(stem_set)] = t.Tensor.tolist(top_activations)
    else:
        stem_set_dict[len(stem_set)].extend(t.Tensor.tolist(top_activations))

unique_stems_dict = {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in sorted(unique_stems_dict.items(), key=lambda x: sum(y[1] for y in x[1]), reverse=True)}

meta_dict = {}

for key, val in stem_set_dict.items():
    average_value = sum(val) / len(val)
    print(f"Average value for key {key}: {average_value}")
    meta_dict[key] = {"avg": average_value, "proportion": (len(val)/10)/N_FEATURES}
meta_dict = dict(sorted(meta_dict.items()))
print(meta_dict)

keys = list(meta_dict.keys())
widths = [entry['proportion'] for entry in meta_dict.values()]
heights = [entry['avg'] for entry in meta_dict.values()]

# Create the bar chart
fig = go.Figure()

for i in range(len(keys)):
    fig.add_trace(
        go.Bar(
            x=[keys[i]],
            y=[heights[i]],
            width=[widths[i] * 5.5],  # Adjusting width for better visualization
            orientation='v'
        )
    )

# Update layout
fig.update_layout(
    title='Proportional Width and Average Activation Height',
    xaxis=dict(title='Number of unique stems in top 10 tokens', tickmode='array', tickvals=list(range(1, 11)), ticktext=list(range(1, 11))),
    yaxis=dict(title='Average Activation of top 10 tokens'),
    bargap=0.05,  # Gap between bars
)

# Show the plot
fig.show()

print(unique_stems)
print(unique_stems/N_FEATURES * 100)

output_file_path = "vibesaurus.json"

with open(output_file_path, "w") as json_file:
    json.dump(unique_stems_dict, json_file)


output_file_path = "vibesaurus_meta.json"

with open(output_file_path, "w") as json_file:
    json.dump(meta_dict, json_file)

# %%
