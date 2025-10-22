import os, numpy as np
from PIL import Image

folder = r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\image-parse-agnostic-v3.2"
for name in ["0000","0001","0007"]:
    p = os.path.join(folder, f"{name}.png")
    im = Image.open(p)
    arr = np.array(im)
    print(name, "mode:", im.mode, "shape:", arr.shape, "unique:", np.unique(arr)[:10])
