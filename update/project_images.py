from pathlib import Path
import requests
import zipfile
import io

import torch
from PIL import Image
import numpy as np

# 1. Obtener la carpeta donde está este .py
BASE_DIR = Path(__file__).resolve().parent.parent

print("Carpeta del script:", BASE_DIR)

# 2. Descargar y descomprimir el dataset en esa carpeta
url = "https://github.com/alexeygrigorev/clothing-dataset-small/archive/refs/heads/master.zip"

response = requests.get(url)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(BASE_DIR)

print("Dataset descargado")

# 3. Buscar automáticamente la carpeta pants
carpetas = list(BASE_DIR.rglob("train/pants"))

if not carpetas:
    raise FileNotFoundError("No se encuentra la carpeta train/pants")

ruta_pants = carpetas[0]

print("Carpeta de imágenes:", ruta_pants)

# 4. Seleccionar una imagen existente
imagenes = list(ruta_pants.glob("*.jpg"))

if not imagenes:
    raise FileNotFoundError("No hay imágenes JPG en pants")

ruta_imagen = imagenes[0]

print("Imagen seleccionada:", ruta_imagen)

# 5. Abrir imagen
img = Image.open(ruta_imagen)

# 6. Redimensionar
img = img.resize((224, 224))

# 7. Convertir a array numpy
x = np.array(img)

print("Dimensiones:", x.shape)

# pytorch
import torchvision.models as models
from torchvision import transforms

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.eval()

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

with torch.no_grad():
    output = model(batch_t)

_, indices = torch.sort(output, descending=True)

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
archivo = "imagenet_classes.txt"

response = requests.get(url)
response.raise_for_status()

with open(archivo, "w", encoding="utf-8") as f:
    f.write(response.text)

print(f"Archivo descargado correctamente: {archivo}")

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

train_path = BASE_DIR / "clothing-dataset-small" / "train"
val_path = BASE_DIR / "clothing-dataset-small" / "validation"

train_dataset = ClothingDataset(data_dir=str(train_path), transform=train_transforms)

val_dataset = ClothingDataset(data_dir=str(val_path), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
