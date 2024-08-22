import torch as t
from transformer_lens import HookedTransformer


def attn_0_derived_features(
    model: HookedTransformer, input_features: t.Tensor
) -> t.Tensor:
    assert input_features.shape[0] == model.cfg.d_model
    WO = model.blocks[0].attn.W_O
    print("WO.shape", WO.shape)
    WV = model.blocks[0].attn.W_V
    print("WV.shape", WV.shape)

    VO = WV @ WO
    print("OV.shape", VO.shape)
    fast_feats = (WV @ WO @ input_features).transpose(-1, -2).flatten(0, 1)
    print("fast_feats.shape", fast_feats.shape)
    return fast_feats.detach()
