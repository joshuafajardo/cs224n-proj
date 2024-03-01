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
    topic_df = pd.read_csv(DATASETS_DIR / "original" / topic_name)
    augmented_df = create_augmented_df(prefixes_df, topic_df, topic_name)
    augmented_df.to_csv(augmented_dir / topic_name)


def create_augmented_df(prefixes_df: pd.DataFrame, topic_df: pd.DataFrame, topic_name: str) -> pd.DataFrame:
  """"""  # TODO
  augmented_df = topic_df.copy()
  augmented_df.rename(columns={"statement": "original_statement"}, inplace=True)
  augmented_df["augmented_statement"] = augmented_df["original_statement"]

  topics_to_decapitalize = {"animals_true_false.csv", "elements_true_false.csv"}
  if topic_name in topics_to_decapitalize:
    augmented_df["augmented_statement"] = augmented_df["augmented_statement"].str.replace(
      r'^[A-Z]',
      lambda match: match.group().lower(),
      regex=True
    )

  augmented_df = prefixes_df.merge(augmented_df, how="cross")
  augmented_df["augmented_statement"] = augmented_df["prefix"] + " " \
                                          + augmented_df["augmented_statement"]
  augmented_df = augmented_df[[
    "augmented_statement",
    "label",
    "prefix",
    "original_statement"
  ]]
  return augmented_df


if __name__ == "__main__":
  main()
