import argparse
import numpy as np
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import get_model, PSNR, L0Loss, UpdateAnnealingParameter
from generator import NoisyImageGenerator, ValGenerator
from noise_model import get_noise_model
from tqdm import tqdm  # 导入 tqdm
from torchvision import transforms

# 定义 transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 统一图像大小为 128x128
    transforms.ToTensor(),          # 转为 Tensor
])




class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=64,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--source_noise_model", type=str, default="gaussian,0,50",
                        help="noise model for source images")
    parser.add_argument("--target_noise_model", type=str, default="gaussian,0,50",
                        help="noise model for target images")
    parser.add_argument("--val_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for validation source images")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    args = parser.parse_args()

    return args









def main():
    args = get_args()

    # 检查是否启用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU acceleration")
    else:
        print("GPU acceleration is not available. Using CPU.")


    #模型选择
    model = get_model(args.model).to(device)


    #断点续训
    if args.weight is not None:
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    #损失函数选择
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    elif args.loss == "l0":
        criterion = L0Loss()
    else:
        raise ValueError("Unknown loss type")





    #噪声模型选择
    #源噪声模型
    source_noise_model = get_noise_model(args.source_noise_model)
    #目标噪声模型
    target_noise_model = get_noise_model(args.target_noise_model)
    #验证噪声模型
    val_noise_model = get_noise_model(args.val_noise_model)
    #训练数据集构造
    train_dataset = NoisyImageGenerator(image_dir=args.image_dir,
                                        source_noise_model=source_noise_model,
                                        target_noise_model=target_noise_model,
                                        batch_size=args.batch_size,
                                        image_size=args.image_size,
                                        transform=transform  # 传入 transform
                                        )
    #得到的4维，又添加了batch_size
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True)





    val_dataset = ValGenerator(image_dir=args.test_dir,
                               val_noise_model=val_noise_model,
                               transform=transform  # 传入 transform
                               )
    val_loader = DataLoader(val_dataset, shuffle=False)

    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def adjust_lr(optimizer, epoch):
        if epoch < args.nb_epochs * 0.25:
            lr = args.lr
        elif epoch < args.nb_epochs * 0.5:
            lr = args.lr * 0.5
        elif epoch < args.nb_epochs * 0.75:
            lr = args.lr * 0.25
        else:
            lr = args.lr * 0.125
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    best_val_psnr = 0

    history = {"train_loss": [], "val_loss": [], "val_psnr": []}



    for epoch in range(args.nb_epochs):
        model.train()
        adjust_lr(optimizer, epoch)

        train_loss = 0
        # 使用 tqdm 添加进度条
        for i, (source_img, target_img) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.nb_epochs}", ncols=100)):
            print(f"Train Batch {i + 1}: Batch size = {source_img.size(0)}")

            source_img = source_img[0, :, :, :, :]  # 去掉第一个拼接用的维度
            target_img = target_img[0, :, :, :, :]  # 去掉第一个拼接用的维度





            source_img = source_img.to(device)
            target_img = target_img.to(device)


            optimizer.zero_grad()


            output = model(source_img)


            loss = criterion(output, target_img)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i + 1 >= args.steps:
                break

        avg_train_loss = train_loss / args.steps

        model.eval()
        val_loss = 0
        psnr_sum = 0
        with torch.no_grad():
            for source_img, target_img in val_loader:

                source_img = source_img.to(device)
                target_img = target_img.to(device)

                output = model(source_img)
                loss = criterion(output, target_img)
                val_loss += loss.item()

                psnr_val = PSNR(output, target_img)
                psnr_sum += psnr_val.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = psnr_sum / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_psnr"].append(avg_val_psnr)
        print(f"Epoch {epoch+1}/{args.nb_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val PSNR: {avg_val_psnr:.4f}")

        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': best_val_psnr
            }, str(output_path / f"weights_epoch{epoch+1:03d}_psnr{avg_val_psnr:.5f}.pth"))




    np.savez(str(output_path.joinpath("history.npz")), history=history)

if __name__ == '__main__':
    main()
