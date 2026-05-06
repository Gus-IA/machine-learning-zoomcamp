from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import subprocess

subprocess.run(
    [
        "git",
        "clone",
        "git clone https://github.com/alexeygrigorev/clothing-dataset-small.git",
    ]
)

img = Image.open(
    "clothing-dataset-small/train/pants/0098b991-e36e-4ef1-b5ee-4154b21e2a92.jpg"
)

img = img.resize((224, 224))
plt.imshow(img)
plt.axis("off")
plt.show()

x = np.array(img)
print(x.shape)


import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np

# Load pre-trained model
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.eval()

# Preprocessing for MobileNetV2
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

x = preprocess(img)
batch_t = torch.unsqueeze(x, 0)
print(x.shape)

# Make prediction
with torch.no_grad():
    output = model(batch_t)
print(output.shape)

# Get top predictions
_, indices = torch.sort(output, descending=True)

import requests

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

response = requests.get(url)
response.raise_for_status()

# Save the content to a file
with open("imagenet_classes.txt", "w") as f:
    f.write(response.text)

# Load ImageNet class names
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Get top 5 predictions
top5_indices = indices[0, :5].tolist()
top5_classes = [categories[i] for i in top5_indices]

print("Top 5 predictions:")
for i, class_name in enumerate(top5_classes):
    print(f"{i+1}: {class_name}")
