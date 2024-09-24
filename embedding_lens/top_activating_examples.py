#%%
import torch as t
from datasets import load_dataset
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import torch

model_name = "gpt2"
model = HookedTransformer.from_pretrained(model_name)
#%%

def tokenize_function(examples) -> dict:
    return model.tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

def get_token_dataloader(dataset_name: str = "stas/openwebtext-10k", split: str = "train", batch_size: int = 16):
    dataset = load_dataset("stas/openwebtext-10k", split="train")

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")

    # Create a DataLoader
    batch_size = 16
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
#%%

def get_top_activating_examples(
    model: t.nn.Module,
    sae: t.nn.Module,
    tl_cache_key: str,
    dataset_name: str = "stas/openwebtext-10k",
    split: str = "train",
    batch_size: int = 16
):
    # Example usage: Iterate through batches
    dataloader = get_token_dataloader(dataset_name, split, batch_size)
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        print("input_ids", input_ids.shape, "attention_mask", attention_mask.shape)

        # Extract number from tl_cache_key
        stop_at_layer = int(tl_cache_key.split(".")[1]) + 1
        with torch.no_grad():
            act, cache = model.run_with_cache(input_ids, attention_mask=attention_mask, stop_at_layer=stop_at_layer)
            recons, latents = sae(cache[tl_cache_key])
        print("recons", recons.shape, "latents", latents.shape)
# %%
