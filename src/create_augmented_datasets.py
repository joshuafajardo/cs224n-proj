# create_augmented_datasets.py

""""""  # TODO: add docstring

import pathlib
import pandas as pd

DATASETS_DIR = pathlib.Path("data/datasets")
PREFIXES_FILE = pathlib.Path("data/prefixes.csv")

TOPIC_NAMES=[
  # "generated_true_false.csv",
  "facts_true_false.csv",  # Did not make the first characters lowercase.
  "animals_true_false.csv",
  "cities_true_false.csv",  # Note: Made some very minor modifications to fix unexpected r"^\s".
  "companies_true_false.csv",
  "elements_true_false.csv",
  "inventions_true_false.csv"
]


def main():
  prefixes_df = pd.read_csv(PREFIXES_FILE)
  augmented_dir = DATASETS_DIR / "augmented"
  augmented_dir.mkdir(parents=True, exist_ok=True)

  for topic_name in TOPIC_NAMES:
    topic_dir = augmented_dir / topic_name
    topic_dir.mkdir(parents=True, exist_ok=True)

    topic_df = pd.read_csv(DATASETS_DIR / "original" / topic_name)
    augmented_dfs = create_augmented_dfs(prefixes_df, topic_df, topic_name)
    for prefix in augmented_dfs:
      augmented_dfs[prefix].to_csv(topic_dir / f"{prefix}")


def create_augmented_dfs(prefixes_df: pd.DataFrame, topic_df: pd.DataFrame, topic_name: str) -> pd.DataFrame:
  """"""  # TODO
  base_augmented_df = topic_df.copy()
  base_augmented_df.rename(columns={"statement": "original_statement"}, inplace=True)
  base_augmented_df["augmented_statement"] = base_augmented_df["original_statement"]

  topics_to_decapitalize = {"animals_true_false.csv", "elements_true_false.csv"}
  if topic_name in topics_to_decapitalize:
    base_augmented_df["augmented_statement"] = \
      base_augmented_df["augmented_statement"].str.replace(
        r'^[A-Z]',
        lambda match: match.group().lower(),
        regex=True
      )

  augmented_dfs = {}
  for prefix in prefixes_df["prefix"]:
    curr_df = base_augmented_df.copy()
    curr_df["augmented_statement"] = prefix + " " + curr_df["augmented_statement"]
    curr_df["prefix"] = prefix
    augmented_dfs[prefix] = curr_df[["augmented_statement",
                                          "label",
                                          "prefix",
                                          "original_statement"]]
  return augmented_dfs

  # augmented_df = prefixes_df.merge(augmented_df, how="cross")
  # augmented_df["augmented_statement"] = augmented_df["prefix"] + " " \
  #                                         + augmented_df["augmented_statement"]
  # augmented_df = augmented_df[[
  #   "augmented_statement",
  #   "label",
  #   "prefix",
  #   "original_statement"
  # ]]
  # return augmented_df


if __name__ == "__main__":
  main()
