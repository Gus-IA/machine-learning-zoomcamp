from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

img = Image.open(
    "clothing-dataset-small/train/pants/0098b991-e36e-4ef1-b5ee-4154b21e2a92.jpg"
)

img = img.resize((224, 224))
plt.imshow(img)
plt.axis("off")
plt.show()

x = np.array(img)
print(x.shape)
