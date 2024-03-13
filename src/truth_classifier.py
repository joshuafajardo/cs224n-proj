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
    self.dropout = nn.Dropout(0.5)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.dropout(self.relu(self.fc1(x)))
    x = self.dropout(self.relu(self.fc2(x)))
    x = self.dropout(self.relu(self.fc3(x)))
    x = self.sigmoid(self.fc4(x))
    return x
