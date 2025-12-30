import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

train = np.load("train.npz")["data"]
test  = np.load("test.npz")["data"]

train_set = {tuple(row) for row in train}
test_set  = {tuple(row) for row in test}

inter = train_set & test_set
print("train ∩ test:", len(inter))
