# get_activations.py

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, MistralForCausalLM

from tqdm import tqdm

from create_augmented_datasets import ORIGINAL_DATASET_DIR, AUGMENTED_DATASET_DIR

ACTIVATIONS_DIR = "data/activations"
LAYERS_TO_SAVE = (16, 20, 24, 28, 32)  # Same as used by Azaria and Mitchell
STATEMENTS_BATCH_SIZE = 8

def main():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
  tokenizer.pad_token = tokenizer.eos_token
  lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)
  print("loaded model")

  for original_csv in glob.glob(f"{ORIGINAL_DATASET_DIR}/*.csv"):
    print(f"Getting activations for {original_csv}")
    df = pd.read_csv(original_csv)
    activations = get_activations(df["statement"], lm, tokenizer, LAYERS_TO_SAVE)
    save_activations(activations, original_csv)
  
  for augmented_csv in glob.glob(f"{AUGMENTED_DATASET_DIR}/*.csv"):
    print(f"Getting activations for {augmented_csv}")
    df = pd.read_csv(activations, augmented_csv)
    activations = get_activations(df["augmented_statement"], lm, tokenizer, LAYERS_TO_SAVE)


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
  for layer in layers:
    activations[layer] = torch.cat(activations[layer], dim=0)  # Concatenate along the batch dimension
  return activations


def save_activations(activations: dict, file_path: str) -> None:
  torch.save(activations, f"{ACTIVATIONS_DIR}/{file_path}.pt")


if __name__ == "__main__":
  main()