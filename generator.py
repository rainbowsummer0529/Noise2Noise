from torchvision import transforms
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import ToPILImage
from PIL import Image




class NoisyImageGenerator(Dataset):
    def __init__(self, image_dir, source_noise_model, target_noise_model, batch_size, image_size, transform=None):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transform  # 保存 transform

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    #返回图片数量
    def __len__(self):
        return self.image_num // self.batch_size


    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_path = random.choice(self.image_paths)
            image = cv2.imread(str(image_path))
            h, w, _ = image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                clean_patch = image[i:i + image_size, j:j + image_size]
                x[sample_id] = self.source_noise_model(clean_patch)
                y[sample_id] = self.target_noise_model(clean_patch)

                # 如果指定了 transform，应用 transform
                if self.transform:
                    pil_x = Image.fromarray(x[sample_id])  # 转为 PIL 图像
                    pil_y = Image.fromarray(y[sample_id])  # 转为 PIL 图像
                    x[sample_id] = self.transform(pil_x).permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                    y[sample_id] = self.transform(pil_y).permute(1, 2, 0).numpy()  # 转换为 (H, W, C)

                sample_id += 1

                if sample_id == batch_size:
                    return torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2), \
                        torch.tensor(y, dtype=torch.float32).permute(0, 3, 1, 2)








from torchvision.transforms import ToPILImage
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

class ValGenerator(Dataset):
    def __init__(self, image_dir, val_noise_model, transform=None):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(image_paths)
        self.data = []
        self.transform = transform

        if self.image_num == 0:
            raise ValueError(f"image dir '{image_dir}' does not include any image")

        # 初始化 ToPILImage 转换器
        self.to_pil = ToPILImage()

        for image_path in image_paths:
            y = cv2.imread(str(image_path))
            h, w, _ = y.shape
            y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            x = val_noise_model(y)

            # 将 NumPy 数组转换为 PIL 图像
            pil_x = self.to_pil(x)
            pil_y = self.to_pil(y)

            # 如果指定了 transform，应用 transform
            if self.transform:
                x = self.transform(pil_x)
                y = self.transform(pil_y)

            # Ensure data is in the correct shape (C, H, W)
            self.data.append([x.unsqueeze(0), y.unsqueeze(0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        x, y = self.data[idx]
        # Ensure the data is in the shape (C, H, W)
        return x.squeeze(0), y.squeeze(0)
