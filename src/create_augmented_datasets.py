# create_augmented_datasets.py

""""""  # TODO: add docstring

import pathlib
import pandas as pd

ORIGINAL_DATASET_DIR = pathlib.Path("data/original_dataset")
AUGMENTED_DATASET_DIR = pathlib.Path("data/augmented_dataset")
PREFIXES_FILE = pathlib.Path("data/prefixes.csv")

TOPIC_NAMES=[
  # "facts_true_false.csv",
  # "generated_true_false.csv",
  "animals_true_false.csv",
  "cities_true_false.csv",  # Note: Made some very minor modifications to fix unexpected r"^\s".
  "companies_true_false.csv",
  "elements_true_false.csv",
  "inventions_true_false.csv"
]


def main():  # TODO: Unsure if we need to remove periods
  prefixes_df = pd.read_csv(PREFIXES_FILE)

  for topic_name in TOPIC_NAMES:
    topic_df = pd.read_csv(ORIGINAL_DATASET_DIR / topic_name)
    augmented_df = create_augmented_df(prefixes_df, topic_df, topic_name)
    augmented_df.to_csv(AUGMENTED_DATASET_DIR / topic_name, index=False)


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
  augmented_df = augmented_df[["augmented_statement", "label", "prefix", "original_statement"]]
  return augmented_df


if __name__ == "__main__":
  main()
