# get_activations.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, MistralForCausalLM

from create_augmented_datasets import ORIGINAL_DATASET_DIR, AUGMENTED_DATASET_DIR, TOPIC_NAMES

STATEMENTS_BATCH_SIZE = 4  # TODO: Find best batch size

def main():
  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
  tokenizer.pad_token = tokenizer.eos_token
  lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
  print("loaded model")

  for topic_name in TOPIC_NAMES:
    print(f"loading {topic_name}")
    df = load_original_df(topic_name)
    print(f"loaded {topic_name}")
    activations = get_activations(df["statement"], lm, tokenizer, [28])
    print(activations)
  
  for topic_name in TOPIC_NAMES:
    df = load_augmented_df(topic_name)
    activations = get_activations(df["augmented_statement"], lm, tokenizer, [28])
    print(activations)


def get_activations(statements: pd.Series,
                    model: LlamaTokenizer,
                    tokenizer: MistralForCausalLM,
                    layers: int) -> torch.Tensor:
  activations = []
  num_batches = np.ceil(len(statements) / STATEMENTS_BATCH_SIZE)
  for batched_statements in np.array_split(statements, num_batches):
    print("batch: ", batched_statements.tolist())
    print("tokenizing batch")
    tokenized_batch = tokenizer(batched_statements.tolist(), padding=True, truncation=True, return_tensors="pt")
    print("tokenized batch")
    for layer in layers:
      print("calling model")
      print(model(**tokenized_batch).hidden_states)
      # activations += model(**tokenized_batch).hidden_states
  # return outputs.hidden_states[layer]

def load_original_df(topic_name) -> pd.DataFrame:
  return pd.read_csv(f"{ORIGINAL_DATASET_DIR}/{topic_name}")


def load_augmented_df(topic_name) -> pd.DataFrame:
  return pd.read_csv(f"{AUGMENTED_DATASET_DIR}/{topic_name}")

if __name__ == "__main__":
  main()