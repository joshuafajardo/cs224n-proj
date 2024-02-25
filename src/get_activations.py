# get_activations.py

import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

SUPPORTED_MODELS = {
  "mistral-7b": "mistralai/Mistral-7B-v0.1"
}


def main():
  tokenizer = AutoTokenizer.from_pretrained(SUPPORTED_MODELS["mistral-7b"])
  lm = AutoModelForCausalLM.from_pretrained(SUPPORTED_MODELS["mistral-7b"])

  augmented_df = pd.read_csv("data/augmented_dataset/animals_true_false.csv")
  augmented_df = augmented_df[["augmented_statement", "label"]]

  for augmented_statement in augmented_df["augmented_statement"]:
    activations = get_activations(augmented_statement)
    print(activations)


def get_activations(tokenized_batch: str, layer: int, model: AutoModelForCausalLM) -> torch.Tensor:
  outputs = model(**tokenized_batch)
  return outputs.hidden_states[layer]


def tokenize_statements(statements: pd.Series, tokenizer: AutoTokenizer) -> torch.Tensor:
  return tokenizer(statements.tolist(), padding=True, return_tensors="pt")


def load_original_df(topic_name) -> pd.DataFrame:
  return pd.read_csv(f"data/original_dataset/{topic_name}")


def load_augmented_df(topic_name) -> pd.DataFrame:
  return pd.read_csv(f"data/augmented_dataset/{topic_name}")