import numpy as np
from pathlib import Path
from flybrain import FlyBrain

# Open the brain file and list every array inside it
data = np.load(Path.home() / "fly-data" / "brain.npz")
for name in data.files:
    arr = data[name]
    print(name, arr.dtype, arr.shape, arr[:5])

# Ask flybrain for the giant fiber escape neurons
brain = FlyBrain(device="cpu")
print("DNp01 cells:", brain.cells(["DNp01"]))
