import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (8, 5)


def get_args():
    parser = argparse.ArgumentParser(description="This script plots training history",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input1", type=str, required=True,
                        help="Path to input checkout directory 1 (must include history.npz)")
    parser.add_argument("--input2", type=str, default=None,
                        help="Path to input checkout directory 2 (must include history.npz) "
                             "if you want to compare it with input1")
    return parser.parse_args()


def main():
    args = get_args()
    input_paths = [Path(args.input1).joinpath("history.npz")]

    if args.input2:
        input_paths.append(Path(args.input2).joinpath("history.npz"))

    datum = [
        (np.array(np.load(str(input_path), allow_pickle=True)["history"], ndmin=1)[0],
         input_path.parent.name)
        for input_path in input_paths
    ]

    metrics = ["val_loss", "val_psnr"]

    for metric in metrics:
        for data, folder_name in datum:
            y = data[metric]
            x = range(1, len(y) + 1)

            # 标签只保留前两个部分，例如 text_noise_unet → text_noise
            label_name = "_".join(folder_name.split("_")[:2])
            plt.plot(x, y, label=label_name)



        plt.xlabel("Epochs")
        plt.ylabel("PSNR" if metric == "val_psnr" else "Loss")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()

        # 修改后的文件名生成方式，只保留第一个文件夹前缀
        prefix = datum[0][1].split("_")[0]
        filename = f"{metric}_{prefix}.png"
        plt.savefig(filename)
        plt.cla()


if __name__ == '__main__':
    main()
