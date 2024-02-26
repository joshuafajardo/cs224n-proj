# trainer.py

import torch
import torch.nn as nn

from tqdm import tqdm


def main():
  pass


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
                              test_loader: torch.utils.data.DataLoader) -> None:
  truth_classifier.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, labels in test_loader:
      outputs = truth_classifier(inputs)
      predicted = (outputs > 0.5).float()
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print(f"Test Accuracy: {correct / total}")


def load_data(data_dir: str) -> torch.utils.data.DataLoader:
  pass