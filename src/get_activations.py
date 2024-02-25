# get_activations.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, MistralForCausalLM

from create_augmented_datasets import ORIGINAL_DATASET_DIR, AUGMENTED_DATASET_DIR, TOPIC_NAMES

STATEMENTS_BATCH_SIZE = 16  # TODO: Find best batch size

def main():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
  tokenizer.pad_token = tokenizer.eos_token
  lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)
  print("loaded model")

  for topic_name in TOPIC_NAMES:
    print(f"loading {topic_name}")
    df = load_original_df(topic_name)
    activations = get_activations(df["statement"], lm, tokenizer, [28])
    print(activations)
  
  for topic_name in TOPIC_NAMES:
    df = load_augmented_df(topic_name)
    activations = get_activations(df["augmented_statement"], lm, tokenizer, [28])
    print(activations)


def get_activations(statements: pd.Series,
                    model: MistralForCausalLM,
                    tokenizer: LlamaTokenizer,
                    layers: int) -> torch.Tensor:
  activations = {layer: [] for layer in layers}
  num_batches = np.ceil(len(statements) / STATEMENTS_BATCH_SIZE)
  for batched_statements in np.array_split(statements, num_batches):
    tokenized_batch = tokenizer(batched_statements.tolist(), padding=True, return_tensors="pt")
    with torch.no_grad():
      hidden_states = model(**tokenized_batch, output_hidden_states=True).hidden_states
    for layer in layers:
      last_token_indices = tokenized_batch["input_ids"].shape[1] - 1
      activations[layer].extend(hidden_states[layer][:, last_token_indices, :].cpu().numpy().tolist())
  return activations

def load_original_df(topic_name) -> pd.DataFrame:
  return pd.read_csv(f"{ORIGINAL_DATASET_DIR}/{topic_name}")


def load_augmented_df(topic_name) -> pd.DataFrame:
  return pd.read_csv(f"{AUGMENTED_DATASET_DIR}/{topic_name}")

if __name__ == "__main__":
  main()