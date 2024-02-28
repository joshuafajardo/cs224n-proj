# trainer.py

import pandas as pd
import torch
import torch.nn as nn

from datetime import datetime
from tqdm import tqdm

from truth_classifier import TruthClassifier
from get_activations import ORIGINAL_ACTIVATIONS_DIR, AUGMENTED_ACTIVATIONS_DIR, LAYERS_TO_SAVE

BATCH_SIZE = 32

def main():
  session_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
  pass

def train_test_each_topic(dataset_type: str):
  """Mainly used for replicating Table 1 of Azaria and Mitchell's paper."""
  match dataset_type:
    case "original":
      activations_dir = ORIGINAL_ACTIVATIONS_DIR
    case "augmented":
      activations_dir = AUGMENTED_ACTIVATIONS_DIR
    case _:
      raise ValueError(f"Invalid dataset type: {dataset_type}")

  topics = {}
  for activation_file in activations_dir.glob("*.pt"):
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

    for layer in LAYERS_TO_SAVE:
      truth_classifier = TruthClassifier()

      train_loader = create_dataloader(train_topics, layer)
      train_truth_classifier(truth_classifier, train_loader)
      train_accuracies[test_topic_name][layer] = evaluate_truth_classifier(
        truth_classifier, train_loader)

      test_loader = create_dataloader([test_topic], layer)
      test_accuracies[test_topic_name][layer] = evaluate_truth_classifier(
        truth_classifier, test_loader)
  
  train_accuracies_df = pd.DataFrame(train_accuracies)
  test_accuracies_df = pd.DataFrame(test_accuracies)
  return train_accuracies_df, test_accuracies_df
  # TODO: save to CSV


def create_dataloader(topics: list[dict], layer) -> torch.utils.data.DataSet:
  inputs = torch.cat([topic["activations"][layer] for topic in topics])
  labels = torch.cat([topic["label"] for topic in topics])
  return torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(inputs, labels),
    batch_size=BATCH_SIZE, shuffle=True)


def train_truth_classifier(truth_classifier: TruthClassifier,
                           loader: torch.utils.data.DataLoader,
                           epochs: int = 5,
                           learning_rate: float = 0.01) -> None:
  loss_func = nn.BCELoss()
  optimizer = torch.optim.Adam(truth_classifier.parameters(), lr=learning_rate)

  for epoch in range(epochs):
    epoch_loss = 0.0
    truth_classifier.train()
    print(f"Beginning epoch {epoch + 1}")
    for inputs, labels in tqdm(loader):
      truth_classifier.zero_grad()
      outputs = truth_classifier(inputs)
      curr_loss = loss_func(outputs, labels)
      curr_loss.backward()
      optimizer.step()
      epoch_loss += curr_loss.item()
    print(f"Epoch {epoch + 1} training loss: {epoch_loss / len(loader)}")


def evaluate_truth_classifier(truth_classifier: TruthClassifier,
                              loader: torch.utils.data.DataLoader) -> None:
  truth_classifier.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, labels in loader:
      outputs = truth_classifier(inputs)
      predictions = (outputs > 0.5).float()
      total += labels.size(0)
      correct += (predictions == labels).sum().item()
  return correct / total
