#%%
from typing import Final, List
from transformer_lens import HookedTransformer
import torch as t
import plotly.graph_objects as go
import plotly.express as px
from torch.nn.functional import mse_loss, cosine_similarity
import os
from datasets import load_dataset

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
# MODEL_NAME: Final[str] = "pythia-2.8b-deduped"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
GATED = True

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
d_model = model.cfg.d_model
embeds, tok_strs = get_embeds(model, DEVICE, en_only=True)

#%%
N_FEATURES = 2000
N_EPOCHS = 4000
L1_LAMBDA = 2e3
LR = 1e-3

TRAIN = True
SAVE = False
LOAD = False

sae_name = "gated_sae" if GATED else "sae"
sae_name = f"{sae_name}_{MODEL_NAME}_{N_FEATURES}_feats_{N_EPOCHS}_epochs_{L1_LAMBDA}_l1_{LR}_lr"
file_name = f"{sae_name}.pth"
file_path = repo_path_to_abs_path(f"trained_saes/{file_name}")

sae = GatedSparseAutoencoder(N_FEATURES, d_model, decoder_bias=embeds.mean(0)).to(DEVICE)
sae.load_state_dict(t.load(file_path))

#%%
WO = model.blocks[0].attn.W_O
print("WO.shape", WO.shape)
WV = model.blocks[0].attn.W_V
print("WV.shape", WV.shape)

VO = WV @ WO
print("OV.shape", VO.shape)
# fast_feats = (sae.encode_weight @ VO).flatten(0, 1)
fast_feats = (WV @ WO @ sae.decode_weight).transpose(-1, -2).flatten(0, 1)
print("fast_feats.shape", fast_feats.shape)

N_FEATURES = 5
with t.no_grad():
    feature_output_logits = embeds @ fast_feats.T
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

#%%
ds=load_dataset('stas/openwebtext-10k')
ds.set_format("torch")
dataloader = t.utils.data.DataLoader(ds["train"], batch_size=32)
#%%
for batch in dataloader:
    toks = model.to_tokens(batch['text'])
   print("toks.shape", toks.shape)
    break