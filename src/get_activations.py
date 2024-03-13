# add_activations.py

import pathlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, MistralForCausalLM

from tqdm import tqdm

from create_augmented_datasets import DATASETS_DIR

ACTIVATIONS_DIR = pathlib.Path("data/activations")
LAYERS_TO_SAVE = (16, 20, 24, 28, 32)  # Same as used by Azaria and Mitchell
STATEMENTS_BATCH_SIZE = 64

def main():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
  tokenizer.pad_token = tokenizer.eos_token
  lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
  lm = lm.to(device)
  print("loaded model")
  
  original_dataset_dir = DATASETS_DIR / "original"
  original_activations_dir = ACTIVATIONS_DIR / "original"
  save_augmented_activations(original_dataset_dir, original_activations_dir,
                             lm, tokenizer, device)

  augmented_dataset_dir = DATASETS_DIR / "augmented"
  augmented_activations_dir = ACTIVATIONS_DIR / "augmented"
  for topic_input_dir in augmented_dataset_dir.glob("*/"):
    topic_name = topic_input_dir.stem
    save_augmented_activations(topic_input_dir,
                               augmented_activations_dir / topic_name, lm,
                               tokenizer, device)


def save_augmented_activations(input_dir: pathlib.Path,
                               output_dir: pathlib.Path,
                               lm: MistralForCausalLM,
                               tokenizer: LlamaTokenizer,
                               device: torch.device) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  for csv_file in input_dir.glob("*.csv"):
    df = pd.read_csv(str(csv_file))
    add_activations(df, lm, tokenizer, device)
    torch.save(df, output_dir / f"{csv_file.stem}.pt")


def add_activations(df: pd.DataFrame,
                    llm: MistralForCausalLM,
                    tokenizer: LlamaTokenizer,
                    device: torch.device) -> dict[int, torch.Tensor]:
  """Add activations to the DataFrame for the given statements."""
  activations = {layer: [] for layer in LAYERS_TO_SAVE}
  if "augmented_statement" in df:
    statements = df["augmented_statement"]
  else:
    statements = df["statement"]

  llm.eval()
  num_batches = np.ceil(len(statements) / STATEMENTS_BATCH_SIZE)
  for batched_statements in tqdm(np.array_split(statements, num_batches)):
    tokenized_batch = tokenizer(batched_statements.tolist(),
                                padding=True,
                                return_tensors="pt")
    tokenized_batch = tokenized_batch.to(device)
    with torch.no_grad():
      hidden_states = \
        llm(**tokenized_batch, output_hidden_states=True).hidden_states

    # For every statement, we only want the activations of the last token.
    statement_indices = torch.arange(len(batched_statements))
    last_token_indices = torch.sum(tokenized_batch.attention_mask, dim=1) - 1
    for layer in LAYERS_TO_SAVE:
      activations[layer].append(
        hidden_states[layer][statement_indices, last_token_indices, :].cpu())

  for layer in LAYERS_TO_SAVE:
    activations[layer] = list(torch.cat(activations[layer], dim=0))
    df[layer_to_colname(layer)] = activations[layer]


def layer_to_colname(layer: int) -> str:
  return f"activations_L{layer}"


if __name__ == "__main__":
  main()