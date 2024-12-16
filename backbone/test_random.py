import numpy as np

split = int(0.1 * 100)
indices = list(range(100))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
print(train_indices)
print(val_indices)