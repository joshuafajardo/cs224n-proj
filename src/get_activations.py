# get_activations.py

import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, MistralForCausalLM

from tqdm import tqdm

from create_augmented_datasets import ORIGINAL_DATASET_DIR, AUGMENTED_DATASET_DIR

ORIGINAL_ACTIVATIONS_DIR = pathlib.Path("data/activations/original")
AUGMENTED_ACTIVATIONS_DIR = pathlib.Path("data/activations/augmented")
LAYERS_TO_SAVE = (16, 20, 24, 28, 32)  # Same as used by Azaria and Mitchell
STATEMENTS_BATCH_SIZE = 1  # TODO: set this back to 8 later

def main():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  ORIGINAL_ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
  AUGMENTED_ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
  tokenizer.pad_token = tokenizer.eos_token
  lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)
  print("loaded model")

  for original_csv in ORIGINAL_DATASET_DIR.glob("*.csv"):
    print(f"Getting activations for {original_csv}")
    df = pd.read_csv(str(original_csv))
    activations = get_activations(df["statement"], lm, tokenizer, LAYERS_TO_SAVE)
    torch.save(activations, ORIGINAL_ACTIVATIONS_DIR / f"{original_csv.stem}.pt")
  
  for augmented_csv in AUGMENTED_DATASET_DIR.glob("*.csv"):
    print(f"Getting activations for {augmented_csv}")
    df = pd.read_csv(activations, str(augmented_csv))
    activations = get_activations(df["augmented_statement"], lm, tokenizer, LAYERS_TO_SAVE)
    torch.save(activations, AUGMENTED_ACTIVATIONS_DIR / f"{augmented_csv.stem}.pt")


def get_activations(statements: pd.Series,
                    model: MistralForCausalLM,
                    tokenizer: LlamaTokenizer,
                    layers: int) -> dict[int, torch.Tensor]:
  activations = {layer: [] for layer in layers}
  num_batches = np.ceil(len(statements) / STATEMENTS_BATCH_SIZE)
  for batched_statements in np.array_split(statements, num_batches):
    tokenized_batch = tokenizer(batched_statements.tolist(), padding=True, return_tensors="pt")
    with torch.no_grad():
      hidden_states = model(**tokenized_batch, output_hidden_states=True).hidden_states
    last_token_indices = tokenized_batch["input_ids"].shape[1] - 1
    for layer in layers:
      activations[layer].append(hidden_states[layer][:, last_token_indices, :].cpu())
    break  # TODO: This is just for debugging.

  for layer in layers:
    activations[layer] = torch.cat(activations[layer], dim=0)  # Concatenate along the batch dimension
  return activations


if __name__ == "__main__":
  main()