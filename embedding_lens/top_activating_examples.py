#%%
import einops
import torch as t
from datasets import load_dataset
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import torch

#%%

def tokenize_function(model, examples) -> dict:
    return model.tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

# def tokenize_function(model, examples, max_length=512):
#     encodings = model.tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
#     return {
#         "input_ids": encodings["input_ids"],
#         "attention_mask": encodings["attention_mask"],
#         "text": examples["text"]
#     }

def get_token_dataloader(model, dataset_name: str = "stas/openwebtext-10k", split: str = "train", batch_size: int = 16):
    dataset = load_dataset("stas/openwebtext-10k", split="train")

    # Tokenize the dataset
    tok_func = lambda x: tokenize_function(model, x)
    tokenized_dataset = dataset.map(tok_func, batched=True, remove_columns=dataset.column_names)
    print("tokenized_dataset", tokenized_dataset)
    tokenized_dataset.set_format("torch")

    # Create a DataLoader
    batch_size = 16
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
#%%

def get_top_activating_examples(
    model: t.nn.Module,
    # sae: t.nn.Module,
    directions: t.Tensor,
    tl_cache_key: str,
    dataset_name: str = "stas/openwebtext-10k",
    split: str = "train",
    batch_size: int = 10
):
    # Example usage: Iterate through batches
    dataloader = get_token_dataloader(model, dataset_name, split, batch_size)
    device = next(model.parameters()).device
    n_directions = directions.shape[0]
    top_dot_prods = t.zeros((n_directions, 100), device=device)
    top_phrases_scores = t.zeros((n_directions, 100, 25), device=device)
    top_phrases_tokens = t.zeros((n_directions, 100, 25), device=device)
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Extract number from tl_cache_key
        stop_at_layer = int(tl_cache_key.split(".")[1]) + 1
        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, attention_mask=attention_mask, stop_at_layer=stop_at_layer)
            new_resids = cache[tl_cache_key]
            flat_resids = einops.rearrange(new_resids, "b s d -> (b s) d")
            del cache, new_resids
            new_dot_prods = einops.einsum(flat_resids, directions, "t d, f d -> t f")
            
            # Take a view of the 12 dot products adjacent to each token in both directions (padding with 0s)
            dot_prods_padded = torch.nn.functional.pad(new_dot_prods, (0, 0, 12, 12), mode='constant', value=0)
            # Use unfold to create sliding windows of size 25
            dot_prod_windows = dot_prods_padded.unfold(dimension=0, size=25, step=1)
            
            # Take a view of the 12 tokens adjacent to each token in both directions (padding with 0s)
            tokens_padded = torch.nn.functional.pad(input_ids, (0, 0, 12, 12), mode='constant', value=0)
            # Use unfold to create sliding windows of size 25
            token_windows = tokens_padded.unfold(dimension=0, size=25, step=1)

            # set top_dot_prods to be the top 100 dot products from the existing top_dot_prods and new_dot_prods for each direction
            top_dot_prods, top_indices = torch.topk(torch.cat([top_dot_prods, new_dot_prods], dim=1), 100, dim=1)
            top_phrases_scores = torch.cat([top_phrases_scores, dot_prod_windows], dim=1)[top_indices]
            top_phrases_tokens = torch.cat([top_phrases_tokens, token_windows], dim=1)[top_indices]

    print("top_dot_prods", top_dot_prods.shape, "top_phrases_scores", top_phrases_scores.shape, "top_phrases_tokens", top_phrases_tokens.shape)
# %%
