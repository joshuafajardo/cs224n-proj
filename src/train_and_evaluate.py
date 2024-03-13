# train_and_evaluate.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import argparse
from datetime import datetime
import pathlib
from tqdm import tqdm

from truth_classifier import TruthClassifier
from get_activations import ACTIVATIONS_DIR, LAYERS_TO_SAVE, layer_to_colname

BASE_RESULTS_DIR = pathlib.Path("results")
TRAINING_BATCH_SIZE = 4096
TEST_BATCH_SIZE = 262144
MISTRAL_HIDDEN_SIZE = 4096
FLOAT_FORMAT = "%.4f"

TRAIN_PREFIXES = [
  "It is true that",
  "It is false that",
  "It must be true that",
  "It must be false that",
  "It should be the case that",
  "It should not be the case that",
  "It could be true that",
  "It's probably not true that",
  "I know it's true that",
  "I know it's false that",
  "I wouldn't doubt that",
  "I highly doubt that",
  "I firmly believe that",
  "I do not believe that",
  "I wouldn't be surprised to find that",
  "I would be shocked to find that",
  "I'm convinced that",
  "I'm not convinced that",
  "I think it's possible that",
  "I think it's unlikely that",
  "Not to be certain, but I think that",
  "Not to be certain, but I doubt that",
  "If I had to guess, I would say that",
  "If I had to guess, I would deny that",
]

TEST_PREFIXES = [
  "I'm 0% certain that",
  "I'm 70% certain that",
  "I'm 90% certain that",
  "I'm 100% certain that",
]

TRAIN_TOPICS = [
  # "generated_true_false",
  "facts_true_false",
  "animals_true_false",
  "cities_true_false",
  "companies_true_false",
  "elements_true_false",
  "inventions_true_false"
]

TEST_TOPICS = [
  "facts_true_false",
  "animals_true_false",
  "cities_true_false",
  "companies_true_false",
  "elements_true_false",
  "inventions_true_false"
]

def main(dataset: str) -> None:
  session_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
  session_dir = BASE_RESULTS_DIR / session_name
  session_dir.mkdir(parents=True, exist_ok=False)

  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  original_result_dir = session_dir / "original"
  augmented_result_dir = session_dir / "augmented"
  both_augmented_result_dir = session_dir / "both_augmented"
  original_result_dir.mkdir(parents=True, exist_ok=True)
  augmented_result_dir.mkdir(parents=True, exist_ok=True)
  both_augmented_result_dir.mkdir(parents=True, exist_ok=True)

  match dataset:
    case "all":
      train_eval_original(original_result_dir, device)
      train_eval_augmented(augmented_result_dir, device)
    case "original":
      train_eval_original(original_result_dir, device)
    case "augmented":
      train_eval_augmented(augmented_result_dir, device)
    case "both_augmented":
      train_eval_both_augmented(
        both_augmented_result_dir,
        TRAIN_TOPICS,
        TRAIN_PREFIXES,
        TEST_TOPICS,
        TEST_PREFIXES,
        device)
    case _:
      raise ValueError("Invalid dataset name")


def train_eval_both_augmented(
    results_dir: pathlib.Path,
    train_topic_names: list[str],
    train_prefixes: list[str],
    test_topic_names: list[str],
    test_prefixes: list[str],
    device: torch.device):

  train_accuracies = {}
  test_accuracies = {}
  average_test_accuracies = {}
  for prefix in test_prefixes:
    average_test_accuracies[prefix] = {}
    for layer in LAYERS_TO_SAVE:
      average_test_accuracies[prefix][layer] = [0, 0]  # (correct, total)

  for test_topic_name in test_topic_names:
    train_accuracies[test_topic_name] = {}
    test_accuracies[test_topic_name] = {}

    curr_train_topic_names = [name for name in train_topic_names if name != test_topic_name]
    curr_train_dfs = []
    for train_topic_name in curr_train_topic_names:
      topic_dfs = []
      for prefix in train_prefixes:
        topic_dfs.append(
          torch.load(ACTIVATIONS_DIR / "augmented" / train_topic_name / f"{prefix}.pt")
        )
      curr_train_dfs.append(topic_dfs)
    
    for prefix in test_prefixes:
      test_accuracies[test_topic_name][prefix] = {}

    for layer in LAYERS_TO_SAVE:
      truth_classifier = TruthClassifier(MISTRAL_HIDDEN_SIZE).to(device)
      num_epochs = 5
      # TODO: Uncomment this later
      # train_loaders = create_sampled_training_dataloaders(
      #   curr_train_dfs, layer, num_dataloaders=num_epochs)
      # train_truth_classifier_multiple_loaders(
      #   truth_classifier, train_loaders, device, learning_rate=10)
      
      all_train_dfs = sum(curr_train_dfs, [])  # Flattens a list of lists
      all_train_dfs_dataloader = create_dataloader(all_train_dfs, layer,
                                                   use_augmented_labels=True)
      train_truth_classifier(
        truth_classifier, all_train_dfs_dataloader, device, epochs=15)  # TODO: Remove later
      train_accuracies[test_topic_name][layer] = evaluate_truth_classifier(
        truth_classifier, all_train_dfs_dataloader, device)
      
      for prefix in test_prefixes:
        test_df = torch.load(
          ACTIVATIONS_DIR / "augmented" / test_topic_name / f"{prefix}.pt")
        test_loader = create_dataloader([test_df], layer,
                                        batch_size=TEST_BATCH_SIZE,
                                        use_augmented_labels=False)
        correct, total = evaluate_truth_classifier(
          truth_classifier, test_loader, device, return_correct_total_counts=True)
        test_accuracies[test_topic_name][prefix][layer] = correct / total
        print("Test accuracy for", test_topic_name, "with prefix", prefix, "at layer", layer, ":", test_accuracies[test_topic_name][prefix][layer])
        average_test_accuracies[prefix][layer][0] += correct
        average_test_accuracies[prefix][layer][1] += total
  
  for prefix in test_prefixes:
    for layer in LAYERS_TO_SAVE:
      average_test_accuracies[prefix][layer] = \
        average_test_accuracies[prefix][layer][0] / average_test_accuracies[prefix][layer][1]
  test_accuracies["average"] = average_test_accuracies
  
  save_dict_to_csv(train_accuracies, results_dir / "train_accuracies.csv")
  for name in test_accuracies:
    save_dict_to_csv(test_accuracies[name],
                     results_dir / f"test_accuracies_{name}.csv")


def train_eval_augmented(
    results_dir: pathlib.Path,
    device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
  # Set up train topics/results
  all_train_topics = {}
  for activation_file in (ACTIVATIONS_DIR / "original").glob("*.pt"):
    all_train_topics[activation_file.stem] = torch.load(activation_file)

  # Set up test topics/results
  all_test_topics = {}
  for topic_dir in (ACTIVATIONS_DIR / "augmented").glob("*/"):
    topic_name = topic_dir.name
    all_test_topics[topic_name] = {}
    for prefix_csv in topic_dir.glob("*.pt"):
      prefix = prefix_csv.stem
      all_test_topics[topic_name][prefix] = torch.load(prefix_csv)

  train_accuracies = {}
  # We intentionally iterate over the test topic names here.
  for name in all_test_topics:
    train_accuracies[name] = {layer: -1 for layer in LAYERS_TO_SAVE}
  
  prefixes = next(iter(all_test_topics.values())).keys()
  test_accuracies = {}
  for name in all_test_topics:
    test_accuracies[name] = {prefix : {} for prefix in prefixes}
    for prefix in prefixes:
      for layer in LAYERS_TO_SAVE:
        test_accuracies[name][prefix][layer] = -1
  

  # Train and evaluate truth classifiers
  for curr_test_topic_name, curr_test_topic in all_test_topics.items():
    curr_train_topics = [all_train_topics[name] for name in all_train_topics if name != curr_test_topic_name]
    for layer in LAYERS_TO_SAVE:
      print(f"Layer: {layer}")
      input_size = curr_train_topics[0][layer_to_colname(layer)][0].shape[0]

      truth_classifier = TruthClassifier(input_size).to(device)
      train_loader = create_dataloader(curr_train_topics, layer)
      train_truth_classifier(
        truth_classifier, train_loader, device)
      train_accuracies[curr_test_topic_name][layer] = [
        evaluate_truth_classifier(truth_classifier, train_loader, device)
      ]
      torch.save(truth_classifier, results_dir / f"classifier_layer{layer}.pt")

      for prefix, df in curr_test_topic:
        print(f"Testing {curr_test_topic_name} with prefix {prefix}")
        test_loader = create_dataloader([df], layer)
        test_accuracies[curr_test_topic_name][prefix][layer] \
          = evaluate_truth_classifier(truth_classifier, test_loader, device)
  
  # Compute test accuracies, averaged over topics
  average_test_accuracies = {}
  all_test_topic_names = list(all_test_topics.keys())  # For consistent ordering
  for prefix in prefixes:
    average_test_accuracies[prefix] = {}
    for layer in LAYERS_TO_SAVE:
      statement_counts = [
        len(all_test_topics[name][prefix]) for name in all_test_topic_names
      ]
      accuracies = [
        test_accuracies[name][prefix][layer] for name in all_test_topic_names
      ]
      average_test_accuracies[prefix][layer] = np.average(
        accuracies, weights=statement_counts)
  test_accuracies["average"] = average_test_accuracies
  
  save_dict_to_csv(train_accuracies, results_dir / "train_accuracies.csv")
  for name in test_accuracies:
    save_dict_to_csv(test_accuracies[name],
                     results_dir / f"test_accuracies_{name}.csv")


def train_eval_original(
    results_dir: pathlib.Path,
    device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Mainly used for replicating Table 1 of Azaria and Mitchell's paper."""
  topics = {}
  for activation_file in (ACTIVATIONS_DIR / "original").glob("*.pt"):
    topics[activation_file.stem] = torch.load(activation_file)
  
  train_accuracies = {}
  test_accuracies = {}
  for name in topics:
    train_accuracies[name] = {layer: -1 for layer in LAYERS_TO_SAVE}
    test_accuracies[name] = {layer: -1 for layer in LAYERS_TO_SAVE}

  # Select one topic at at time for testing, and use the rest for training.
  for test_topic_name in topics:
    test_topic = topics[test_topic_name]
    train_topics = [topics[name] for name in topics if name != test_topic_name]
    print("Topic: ", test_topic_name)

    for layer in LAYERS_TO_SAVE:
      print(f"Layer: {layer}")
      input_size = train_topics[0][layer_to_colname(layer)][0].shape[0]
      truth_classifier = TruthClassifier(input_size).to(device)

      train_loader = create_dataloader(train_topics, layer)
      train_truth_classifier(truth_classifier, train_loader, device)
      train_accuracies[test_topic_name][layer] = evaluate_truth_classifier(
        truth_classifier, train_loader, device)

      test_loader = create_dataloader([test_topic], layer)
      test_accuracies[test_topic_name][layer] = evaluate_truth_classifier(
        truth_classifier, test_loader, device)

      torch.save(truth_classifier,
                 results_dir / f"classifier_{test_topic_name}_layer{layer}.pt")
  
  # Compute average accuracies
  average_train_accuracies = {}
  average_test_accuracies = {}
  topic_names = list(topics.keys())  # For consistent ordering
  statement_counts = [len(topics[name]) for name in topic_names]
  for layer in LAYERS_TO_SAVE:
    average_train_accuracies[layer] = np.average(
      [train_accuracies[name][layer] for name in topic_names],
      weights=statement_counts)
    average_test_accuracies[layer] = np.average(
      [test_accuracies[name][layer] for name in topic_names],
      weights=statement_counts)
  train_accuracies["average"] = average_train_accuracies
  test_accuracies["average"] = average_test_accuracies
  

  save_dict_to_csv(train_accuracies, results_dir / "train_accuracies.csv")
  save_dict_to_csv(test_accuracies, results_dir / "test_accuracies.csv")


def create_sampled_training_dataloaders(
    topics: list[list[pd.DataFrame]],  # List of topics, each containing a list of dataframes for each prefix
    layer: int,
    num_dataloaders: int) -> dict[int, torch.utils.data.Dataset]:
  dataloaders = []
  for i in range(num_dataloaders):
    inputs = []
    labels = []
    for topic in topics:
      topic_length = len(topic[0])
      num_prefixes = len(topic)
      samples = np.random.randint(0, high=num_prefixes, size=topic_length)
      for prefix_num in range(num_prefixes):
        chosen_rows = topic[prefix_num][prefix_num == samples]
        inputs.append(torch.stack(list(chosen_rows[layer_to_colname(layer)])))
        labels.append(torch.tensor(chosen_rows["augmented_label"].values))
    inputs = torch.cat(inputs)
    labels = torch.cat(labels).unsqueeze(1).float()
    dataloaders.append(torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(inputs, labels),
      batch_size=TRAINING_BATCH_SIZE, shuffle=True
    ))
  return dataloaders

    

def create_dataloader(
    topics: list[pd.DataFrame],
    layer: int,
    batch_size = TRAINING_BATCH_SIZE,
    use_augmented_labels: bool = False) -> torch.utils.data.Dataset:
  inputs = pd.concat([topic[layer_to_colname(layer)] for topic in topics])
  inputs = torch.stack(list(inputs.values))
  labels_to_use = "augmented_label" if use_augmented_labels else "original_label"
  labels = torch.cat(
    [torch.tensor(topic[labels_to_use].values) for topic in topics])
  labels = labels.unsqueeze(1).float()

  return torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(inputs, labels),
    batch_size=batch_size, shuffle=True)


def train_truth_classifier_multiple_loaders(
    truth_classifier: TruthClassifier,
    loaders: list[torch.utils.data.DataLoader],
    device: torch.device,
    learning_rate: float = 0.001) -> None:
  truth_classifier.to(device)
  truth_classifier.train()
  optimizer = torch.optim.Adam(truth_classifier.parameters(), lr=learning_rate)
  loss_func = nn.BCELoss()

  for epoch_num in range(len(loaders)):
    epoch_loss = 0.0
    loader = loaders[epoch_num]
    for inputs, labels in loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      truth_classifier.zero_grad()
      outputs = truth_classifier(inputs)
      curr_loss = loss_func(outputs, labels)
      curr_loss.backward()
      optimizer.step()
      epoch_loss += curr_loss.item()
    print(f"Epoch {epoch_num + 1} training loss: {epoch_loss / len(loader)}")


def train_truth_classifier(truth_classifier: TruthClassifier,
                           loader: torch.utils.data.DataLoader,
                           device: torch.device,
                           epochs: int = 5,
                           learning_rate: float = 0.001) -> None:
  truth_classifier.to(device)
  truth_classifier.train()
  optimizer = torch.optim.Adam(truth_classifier.parameters(), lr=learning_rate)
  loss_func = nn.BCELoss()

  for epoch in range(epochs):
    epoch_loss = 0.0
    for inputs, labels in loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      truth_classifier.zero_grad()
      outputs = truth_classifier(inputs)
      curr_loss = loss_func(outputs, labels)
      curr_loss.backward()
      optimizer.step()
      epoch_loss += curr_loss.item()
    print(f"Epoch {epoch + 1} training loss: {epoch_loss / len(loader)}")


def evaluate_truth_classifier(truth_classifier: TruthClassifier,
                              loader: torch.utils.data.DataLoader,
                              device: torch.device,
                              return_correct_total_counts=False) -> None:
  truth_classifier.to(device)
  truth_classifier.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, labels in loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = truth_classifier(inputs)
      predictions = (outputs > 0.5).float()
      total += labels.size(0)
      correct += (predictions == labels).sum().item()
  if return_correct_total_counts:
    return correct, total
  else:
    return correct / total


def save_dict_to_csv(data: dict, file_path: pathlib.Path) -> None:
  df = pd.DataFrame(data)
  df.to_csv(file_path, float_format=FLOAT_FORMAT)



if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(
    prog="train_and_evaluate.py",
    description="Train and evaluate various truth classifiers.")
  arg_parser.add_argument("-d", "--dataset", type=str, default="all",
                          help="Dataset to use: all, original, augmented, or " +
                            "both_augmented")
  args = arg_parser.parse_args()
  main(args.dataset)
