# truth_classifier.py

import torch
import torch.nn as nn

class TruthClassifier(nn.Module):
  def __init__(self, input_size: int):
    super(TruthClassifier, self).__init__()
    self.fc1 = nn.Linear(input_size, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)
    self.fc4 = nn.Linear(64, 1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.sigmoid(self.fc4(x))
    print(x.shape)
    return x


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