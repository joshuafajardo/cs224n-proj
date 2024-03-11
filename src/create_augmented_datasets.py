# create_augmented_datasets.py

""""""  # TODO: add docstring

import pathlib
import pandas as pd

DATASETS_DIR = pathlib.Path("data/datasets")
PREFIXES_FILE = pathlib.Path("data/prefixes.csv")

TOPIC_NAMES=[
  # "generated_true_false",
  "facts_true_false",  # Did not make the first characters lowercase.
  "animals_true_false",
  "cities_true_false",  # Note: Made some very minor modifications to fix unexpected r"^\s".
  "companies_true_false",
  "elements_true_false",
  "inventions_true_false"
]


def main():
  prefixes_df = pd.read_csv(PREFIXES_FILE)
  augmented_dir = DATASETS_DIR / "augmented"
  augmented_dir.mkdir(parents=True, exist_ok=True)

  for topic_name in TOPIC_NAMES:
    output_dir = augmented_dir / topic_name
    output_dir.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(DATASETS_DIR / "original" / f"{topic_name}.csv")
    augmented_dfs = create_augmented_dfs(prefixes_df, original_df, topic_name)
    for prefix in augmented_dfs:
      augmented_dfs[prefix].to_csv(output_dir / f"{prefix}.csv")


def create_augmented_dfs(prefixes_df: pd.DataFrame, original_df: pd.DataFrame, topic_name: str) -> pd.DataFrame:
  base_augmented_df = original_df.copy()
  base_augmented_df.rename(columns={"statement": "original_statement",
                                    "label": "original_label"}, inplace=True)
  base_augmented_df["augmented_statement"] = base_augmented_df["original_statement"]

  topics_to_decapitalize = {"animals_true_false", "elements_true_false"}
  if topic_name in topics_to_decapitalize:
    base_augmented_df["augmented_statement"] = \
      base_augmented_df["augmented_statement"].str.replace(
        r'^[A-Z]',
        lambda match: match.group().lower(),
        regex=True
      )

  augmented_dfs = {}
  for _, row in prefixes_df.iterrows():
    prefix = row["prefix"]
    affirms = row["affirms"]
    curr_df = base_augmented_df.copy()
    curr_df["augmented_statement"] = prefix + " " + curr_df["augmented_statement"]
    curr_df["augmented_label"] = int((curr_df["original_label"] - affirms) % 2)
    curr_df["prefix"] = prefix
    augmented_dfs[prefix] = curr_df[[
      "augmented_statement",
      "augmented_label",
      "prefix",
      "original_statement",
      "original_label"
    ]]
  return augmented_dfs


if __name__ == "__main__":
  main()
