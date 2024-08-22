# %%
import torch as t
from transformer_lens import HookedTransformer
from transformers import GPT2LMHeadModel


def no_layernorm_gpt2(device) -> HookedTransformer:
    model = GPT2LMHeadModel.from_pretrained("apollo-research/gpt2_noLN")

    # Undo my hacky LayerNorm removal
    for block in model.transformer.h:
        block.ln_1.weight.data = block.ln_1.weight.data / 1e6
        block.ln_1.eps = 1e-5
        block.ln_2.weight.data = block.ln_2.weight.data / 1e6
        block.ln_2.eps = 1e-5
    model.transformer.ln_f.weight.data = model.transformer.ln_f.weight.data / 1e6
    model.transformer.ln_f.eps = 1e-5

    # Properly replace LayerNorms by Identities
    def removeLN(tl_model: HookedTransformer):
        for i in range(len(tl_model.blocks)):
            tl_model.blocks[i].ln1 = t.nn.Identity()
            tl_model.blocks[i].ln2 = t.nn.Identity()
        tl_model.ln_final = t.nn.Identity()

    model: HookedTransformer = HookedTransformer.from_pretrained(
        "gpt2", hf_model=model, fold_ln=True, center_unembed=False, device=device
    )
    removeLN(model)
    return model


# %%
# DEVICE = "cuda:2" if t.cuda.is_available() else "cpu"
# model = no_layernorm_gpt2(DEVICE)
# model.generate("An interpretable architecture can only see widespread", max_new_tokens=100, do_sample=True, temperature=0.8, prepend_bos=True)
