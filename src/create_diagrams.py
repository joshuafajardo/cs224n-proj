import matplotlib.pyplot as plt
import numpy as np

layers = [16, 20, 24, 28, 32]
prefixes = ["0%", "70%", "90%", "100%"]

test_accuracies = {
  5: np.array([
    [0.4983, 0.5265, 0.5280, 0.5241],
    [0.4953, 0.5177, 0.5193, 0.5180],
    [0.4978, 0.5150, 0.5136, 0.5196],
    [0.4994, 0.5086, 0.5073, 0.5103],
    [0.4984, 0.5048, 0.5055, 0.5022],
  ]),
  20: np.array([
    [0.5063, 0.5540, 0.5518, 0.5486],
    [0.4917, 0.5224, 0.5205, 0.5168],
    [0.4991, 0.5173, 0.5118, 0.5122],
    [0.5104, 0.5152, 0.5205, 0.5159],
    [0.4960, 0.5121, 0.5147, 0.5101],
  ]),
}

vmin = min([np.min(test_accuracies[5]), np.min(test_accuracies[20])])
vmax = max([np.max(test_accuracies[5]), np.max(test_accuracies[20])])

figure, (ax5, ax20) = plt.subplots(1, 2)
im5 = ax5.imshow(test_accuracies[5], vmin=vmin, vmax=vmax)
im20 = ax20.imshow(test_accuracies[20], vmin=vmin, vmax=vmax)

ax5.set_xticks(np.arange(len(prefixes)), labels=prefixes)
ax20.set_xticks(np.arange(len(prefixes)), labels=prefixes)

ax5.set_yticks(np.arange(len(layers)), labels=layers)
ax20.set_yticks(np.arange(len(layers)), labels=layers)

for i in range(len(layers)):
  for j in range(len(prefixes)):
    ax5.text(j, i, test_accuracies[5][i, j], ha="center", va="center", color="w")
    ax20.text(j, i, test_accuracies[20][i, j], ha="center", va="center", color="w")

figure.colorbar(im5, ax=[ax5, ax20])

ax5.set_title("Test Accuracies, 5 Epochs")
ax20.set_title("Test Acccuracies, 20 Epochs")

ax5.set_xlabel("Numerical Value in Prefix: \"I am X% certain that\"")
ax20.set_xlabel("Numerical Value in Prefix \"I am X% certain that\"")

ax5.set_ylabel("LLM Layer Number")
ax20.set_ylabel("LLM Layer Number")


plt.show()