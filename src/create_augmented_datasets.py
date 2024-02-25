# create_augmented_datasets.py

""""""  # TODO: add docstring

import pandas as pd

ORIGINAL_DATASET_DIR = "data/original_dataset"
AUGMENTED_DATASET_DIR = "data/augmented_dataset"
PREFIXES_FILE = "data/prefixes.csv"

TOPIC_NAMES=[
  # "facts_true_false.csv",
  # "generated_true_false.csv",
  "animals_true_false.csv",
  "cities_true_false.csv",  # Note: Made some very minor modifications to fix unexpected r"^\s".
  "companies_true_false.csv",
  "elements_true_false.csv",
  "inventions_true_false.csv"
]


def main():
  prefixes = load_prefixes()

  for topic_name in TOPIC_NAMES:
    topic_df = pd.read_csv(f"{ORIGINAL_DATASET_DIR}/{topic_name}")
    augmented_df = add_prefixes(prefixes, topic_df, topic_name)
    augmented_df = augmented_df[["augmented_statement", "label", "prefix", "statement"]]
    save_df_to_csv(augmented_df, f"{AUGMENTED_DATASET_DIR}/{topic_name}")

def add_prefixes(prefixes: pd.DataFrame, topic_df: pd.DataFrame, topic_name: str) -> pd.DataFrame:
  """"""  # TODO
  augmented_df = topic_df.copy()
  augmented_df["augmented_statement"] = topic_df["statement"]

  topics_to_decapitalize = {"animals_true_false.csv", "elements_true_false.csv"}
  if topic_name in topics_to_decapitalize:
    augmented_df["augmented_statement"] = augmented_df["augmented_statement"].str.replace(
      r'^[A-Z]',
      lambda match: match.group().lower(),
      regex=True
    )

  augmented_df = prefixes.merge(augmented_df, how="cross")
  augmented_df["augmented_statement"] = augmented_df["prefix"] + " " \
                                          + augmented_df["augmented_statement"]
  return augmented_df


def load_prefixes() -> pd.DataFrame:
  """"""  # TODO
  return pd.read_csv(PREFIXES_FILE)


def save_df_to_csv(df: pd.DataFrame, file_path: str) -> None:
  """"""  # TODO
  df.to_csv(file_path, index=False)


if __name__ == "__main__":
  main()
