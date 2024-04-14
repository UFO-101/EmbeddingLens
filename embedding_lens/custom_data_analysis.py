from typing import Final

from spellchecker import SpellChecker
from embedding_lens.sae import SparseAutoencoder
from embedding_lens.utils import repo_path_to_abs_path
from transformer_lens import HookedTransformer
import torch as t

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
tok_strs: List[str] = model.to_str_tokens(t.arange(model.cfg.d_vocab))  # type: ignore
tok_strs = [word.strip() for word in tok_strs]
correct_words = spell.known(tok_strs)
en_idxs = [i for i, tok_str in enumerate(tok_strs) if tok_str.strip() in correct_words]
print("embeds", embeds.shape)
embeds = embeds[en_idxs]
print("en embeds", embeds.shape)

file_name = f"sae_{MODEL_NAME}_{N_FEATURES}_feats_{N_EPOCHS}_epochs_{L1_LAMBDA}_l1_{LR}_lr.pth"
file_path = repo_path_to_abs_path(f"trained_saes/{file_name}")

sae = SparseAutoencoder(N_FEATURES, d_model, decoder_bias=embeds.mean(0)).to(DEVICE)
sae.load_state_dict(t.load(file_path))

with t.no_grad():
    recons, l1s, latents = sae(embeds)
