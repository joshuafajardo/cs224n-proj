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
  inputs = tokenizer(augmented_statements, return_tensors="pt")
  outputs = model(**inputs)
  return outputs.hidden_states[layer]


def load_topic(topic_name: str) -> pd.DataFrame:
  return pd.read_csv(f"data/augmented_dataset/{topic_name}")