from typing import List, Tuple
from spellchecker import SpellChecker
from transformer_lens import HookedTransformer
import torch as t

spell = SpellChecker()

def get_embeds(
    model: HookedTransformer,
    device: str,
    en_only: bool
) -> Tuple[t.Tensor, List[str]]:
    embeds = model.W_E.detach().clone().to(device)
    tok_strs: List[str] = model.to_str_tokens(t.arange(model.cfg.d_vocab))  # type: ignore

    #%%
    # filter to english words
    if en_only:
        tok_strs = [word.strip() for word in tok_strs]
        correct_words = spell.known(tok_strs)
        en_idxs = [i for i, tok_str in enumerate(tok_strs) if tok_str.strip() in correct_words]
        tok_strs = [tok_str for tok_str in tok_strs if tok_str.strip() in correct_words]
        print("embeds", embeds.shape)
        embeds = embeds[en_idxs]
        print("en embeds", embeds.shape)
    return embeds, tok_strs