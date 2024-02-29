# add_activations.py

import pathlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, MistralForCausalLM

from tqdm import tqdm

from create_augmented_datasets import ORIGINAL_DATASET_DIR, AUGMENTED_DATASET_DIR

ORIGINAL_ACTIVATIONS_DIR = pathlib.Path("data/activations_mistral-7b/original")
AUGMENTED_ACTIVATIONS_DIR = pathlib.Path("data/activations_mistral-7b/augmented")
LAYERS_TO_SAVE = (16, 20, 24, 28, 32)  # Same as used by Azaria and Mitchell
STATEMENTS_BATCH_SIZE = 4

def main():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  ORIGINAL_ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
  AUGMENTED_ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
  tokenizer.pad_token = tokenizer.eos_token
  lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
  lm = lm.to(device)
  print("loaded model")

  for original_csv in ORIGINAL_DATASET_DIR.glob("*.csv"):
    print(f"Getting activations for {original_csv}")
    df = pd.read_csv(str(original_csv))
    add_activations(df, lm, tokenizer, LAYERS_TO_SAVE, device)
    torch.save(df, ORIGINAL_ACTIVATIONS_DIR / f"{original_csv.stem}.pt")
  
  for augmented_csv in AUGMENTED_DATASET_DIR.glob("*.csv"):
    print(f"Getting activations for {augmented_csv}")
    df = pd.read_csv(str(augmented_csv))
    add_activations(df, lm, tokenizer, LAYERS_TO_SAVE, device)
    torch.save(df, AUGMENTED_ACTIVATIONS_DIR / f"{augmented_csv.stem}.pt")


def add_activations(df: pd.DataFrame,
                    model: MistralForCausalLM,
                    tokenizer: LlamaTokenizer,
                    layers: int,
                    device: torch.device) -> dict[int, torch.Tensor]:
  """Add activations to the DataFrame for the given statements."""
  activations = {layer: [] for layer in layers}
  if "augmented_statement" in df:
    statements = df["augmented_statement"]
  else:
    statements = df["statement"]

  num_batches = np.ceil(len(statements) / STATEMENTS_BATCH_SIZE)
  for batched_statements in tqdm(np.array_split(statements, num_batches)):
    tokenized_batch = tokenizer(batched_statements.tolist(), padding=True, return_tensors="pt")
    tokenized_batch = tokenized_batch.to(device)
    with torch.no_grad():
      hidden_states = model(**tokenized_batch, output_hidden_states=True).hidden_states
    last_token_indices = tokenized_batch["input_ids"].shape[1] - 1
    for layer in layers:
      print(f"hidden states: {hidden_states[layer][:, last_token_indices, :].cpu()}")
      print(f"hidden states: {hidden_states[layer].cpu()}")
      activations[layer].append(hidden_states[layer][:, last_token_indices, :].cpu())

  for layer in layers:
    activations[layer] = torch.cat(activations[layer], dim=0)  # Concatenate along the batch dimension
    print(f"layer {layer} activations: {activations[layer]}")
  print(activations)
  df["activations"] = activations


if __name__ == "__main__":
  main()