from pathlib import Path
import requests
import zipfile
import io

import torch
from PIL import Image
import numpy as np

# 1. Obtener la carpeta donde está este .py
BASE_DIR = Path(__file__).resolve().parent

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
