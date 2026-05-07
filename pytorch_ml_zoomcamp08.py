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


import os
from torch.utils.data import Dataset
from PIL import Image


# Custom dataset class
class ClothingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


from torchvision import transforms

# Preprocessing

input_size = 224

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Simple transforms - just resize and normalize
train_transforms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


from torch.utils.data import DataLoader

# Dataloader

train_dataset = ClothingDataset(
    data_dir="./clothing-dataset-small/train", transform=train_transforms
)

val_dataset = ClothingDataset(
    data_dir="./clothing-dataset-small/validation", transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
