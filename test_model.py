import argparse
import numpy as np
from pathlib import Path
import cv2
import torch
from model import get_model
from noise_model import get_noise_model
import os

def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    val_noise_model = get_noise_model(args.test_noise_model)

    # 创建模型
    model = get_model(args.model)


    # 加载权重
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 切换到评估模式

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有图像
    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape
        image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
        h, w, _ = image.shape

        # 将图像转换为 PyTorch tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 为了去噪，将噪声模型应用到图像
        noise_image = val_noise_model(image)
        noise_image_tensor = torch.from_numpy(noise_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 推理，得到预测结果
        with torch.no_grad():
            pred = model(noise_image_tensor)

        # 将结果转回 numpy 并去除多余的维度
        denoised_image = get_image(pred.squeeze(0).permute(1, 2, 0).numpy() * 255)

        # 组合原图、噪声图和去噪图
        out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
        out_image[:, :w] = image
        out_image[:, w:w * 2] = noise_image
        out_image[:, w * 2:] = denoised_image

        # 输出结果
        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
        else:
            cv2.imshow("result", out_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()





