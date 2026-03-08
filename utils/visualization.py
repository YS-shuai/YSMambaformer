import os

from matplotlib import pyplot as plt
from torch import nn
from torchvision.utils import save_image


class vision(nn.Module):
    def __init__(
        self, setting, context_length, target_length, save_dir
    ):
        super().__init__()
        self.context_length = context_length
        self.target_length = target_length
        self.setting = setting
        self.save_dir = save_dir

    def forward(self, preds, batch):

        targ = batch["dynamic"][0].detach().cpu()
        pred = preds[0].detach().cpu()
        if self.setting not in ["en21x", "en21_list", "DynamicNet7M"]:
            targ_save_dir = os.path.join(self.save_dir, "target_sequence")
            pred_save_dir = os.path.join(self.save_dir, "pred_sequence")
            if not os.path.exists(targ_save_dir):
                os.makedirs(targ_save_dir)
            if not os.path.exists(pred_save_dir):
                os.makedirs(pred_save_dir)

            for t in range(targ.shape[0]):
                # 获取当前时间点的图像
                img = targ[t]

                # 判断通道数，确定保存图像的方式
                if img.shape[0] == 3:  # C == 3, RGB图像
                    # 将归一化后的图像数据乘以255，转换回[0, 255]范围
                    img = (img * 255).byte()
                    # 保存为RGB图像
                    save_image(img, f"{targ_save_dir}/target_{t}.png")
                elif img.shape[0] == 1:  # C == 1, 灰度图像
                    # 将归一化后的图像数据乘以255，转换回[0, 255]范围
                    img = (img[0] * 255).byte()  # 只取第一个通道
                    # 保存为灰度图像
                    plt.imsave(f"{targ_save_dir}/target_{t}.png", img, cmap='gray')
                else:
                    raise ValueError("Unsupported number of channels. Only 1 or 3 channels are supported.")

                for t in range(pred.shape[0]):
                    # 获取当前时间点的图像
                    img = pred[t]

                    # 判断通道数，确定保存图像的方式
                    if img.shape[0] == 3:  # C == 3, RGB图像
                        # 将归一化后的图像数据乘以255，转换回[0, 255]范围
                        img = (img * 255).byte()
                        # 保存为RGB图像
                        save_image(img, f"{pred_save_dir}/pred_{t+self.context_length}.png")
                    elif img.shape[0] == 1:  # C == 1, 灰度图像
                        # 将归一化后的图像数据乘以255，转换回[0, 255]范围
                        img = (img[0] * 255).byte()  # 只取第一个通道
                        # 保存为灰度图像
                        plt.imsave(f"{pred_save_dir}/pred_{t+self.context_length}.png", img, cmap='gray')
                    else:
                        raise ValueError("Unsupported number of channels. Only 1 or 3 channels are supported.")



