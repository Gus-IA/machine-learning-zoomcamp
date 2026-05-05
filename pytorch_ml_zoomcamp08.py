import subprocess

subprocess.run(
    ["git", "clone", "https://github.com/alexeygrigorev/clothing-dataset-small.git"]
)

import torch
from PIL import Image

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

img = Image.open(
    "clothing-dataset-small/train/pants/0a7e5fe0-d592-40e6-b9b8-75aac9a2d685.jpg"
)

img = img.resize((224, 224))
x = np.array(img)
print(x.shape)
plt.imshow(img)
plt.axis("off")
plt.show()
