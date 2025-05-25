import torch
import torch.nn as nn
import torch.nn.functional as F


# L0损失
class L0Loss(nn.Module):
    def __init__(self):
        super(L0Loss, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(2.0))

    def forward(self, y_pred, y_true):
        loss = torch.abs(y_true - y_pred) + 1e-8
        loss = loss ** self.gamma
        return loss.mean()  # ← 加上这一句，让返回的是标量



class UpdateAnnealingParameter:
    def __init__(self, gamma_tensor, nb_epochs, verbose=0):
        self.gamma_tensor = gamma_tensor  # nn.Parameter from L0Loss
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def update(self, epoch):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        self.gamma_tensor.data.fill_(new_gamma)
        if self.verbose > 0:
            print(f"Epoch {epoch + 1}: Updated gamma to {new_gamma:.4f}")








#计算PSNR 值越大效果越好
def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = torch.clamp(y_pred, 0.0, 255.0)
    mse = torch.mean((y_pred.float() - y_true.float()) ** 2)
    psnr = 10.0 * torch.log10((max_pixel ** 2) / (mse + 1e-8))
    return psnr




#选择模型
def get_model(model_name="srresnet"):
    if model_name == "srresnet":
        return get_srresnet_model()
    elif model_name == "unet":
        return get_unet_model(out_ch=3)
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")


# 残差块模块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual


# 主模型函数：保持与 Keras 同名
class get_srresnet_model(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64, resunit_num=16):
        super(get_srresnet_model, self).__init__()

        # 初始卷积 + PReLU
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channel_num, feature_dim, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # 残差块序列
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(feature_dim) for _ in range(resunit_num)]
        )

        # 中间卷积 + BN
        self.mid_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim)
        )

        # 输出层
        self.output_conv = nn.Conv2d(feature_dim, input_channel_num, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.input_conv(x)
        x = self.res_blocks(x0)
        x = self.mid_conv(x)
        x = x + x0  # 残差连接
        x = self.output_conv(x)
        return x


#卷积块模型
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu', batchnorm=False, dropout=0, residual=False):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()

    def forward(self, x):


        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.activation(self.bn2(self.conv2(out)))


        # if self.residual:
        #     # 拼接输入和输出通道数，保持尺寸一致，需要输入通道数==输出通道数
        #     out = torch.cat([x, out], dim=1)

        return out




class UNetLevel(nn.Module):
    def __init__(self, in_ch, out_ch, depth, inc_rate, activation, dropout,
                 batchnorm, maxpool, upconv, residual):
        super(UNetLevel, self).__init__()
        self.depth = depth
        self.maxpool = maxpool
        self.upconv = upconv
        self.residual = residual
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.inc_rate = inc_rate
        self.activation = activation

        self.conv_block = ConvBlock(in_ch, out_ch, activation, batchnorm, dropout if depth == 0 else 0, residual)
        if depth > 0:
            if maxpool:
                self.downsample = nn.MaxPool2d(2)
            else:
                self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            self.sub_level = UNetLevel(out_ch, int(out_ch * inc_rate), depth - 1, inc_rate, activation,
                                       dropout, batchnorm, maxpool, upconv, residual)
            if upconv:
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(int(out_ch * inc_rate), out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
                )
            else:
                self.upsample = nn.ConvTranspose2d(int(out_ch * inc_rate), out_ch, kernel_size=3, stride=2, padding=1,
                                                   output_padding=1)
            self.conv_block2 = ConvBlock(out_ch * 2, out_ch, activation, batchnorm, 0, residual)




    def forward(self, x):
        n = self.conv_block(x)
        if self.depth > 0:
            m = self.downsample(n)
            m = self.sub_level(m)
            m = self.upsample(m)
            # 拼接 skip connection


            m = torch.cat([n, m], dim=1)

            m = self.conv_block2(m)
        else:
            m = n
        return m


def get_unet_model(input_channel_num=3, out_ch=3, start_ch=64, depth=4, inc_rate=2.,
                   activation='relu', dropout=0.5, batchnorm=False,
                   maxpool=True, upconv=True, residual=False):
    model = UNetLevel(input_channel_num, start_ch, depth, inc_rate, activation,
                      dropout, batchnorm, maxpool, upconv, residual)
    final_conv = nn.Conv2d(start_ch, out_ch, kernel_size=1)


    class UNetWrapper(nn.Module):
        def __init__(self, model, final_conv):
            super(UNetWrapper, self).__init__()
            self.model = model
            self.final_conv = final_conv

        def forward(self, x):
            x = self.model(x)
            x = self.final_conv(x)
            return x

    return UNetWrapper(model, final_conv)


def main():
    model = get_unet_model()
    print(model)  # 打印模型结构

    # 你可以打印模型参数数量，类似 summary 的简单信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

if __name__ == '__main__':
    main()
