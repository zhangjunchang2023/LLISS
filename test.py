import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import sys
import matplotlib

from miou import IoUMetric

matplotlib.use('Agg')
import utils
import models
from dataset import LLRGBD_real

# 设置打印选项
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile='full')
torch.backends.cudnn.benchmark = True



def test(args, Decomp, Enhance, testloader):
    print('Start Testing...')
    Decomp.eval()
    Enhance.eval()

    with torch.no_grad():
        # 初始化评估器
        metric = IoUMetric(num_classes=14, device='cuda')
        for i, (image, image2, label, name) in enumerate(tqdm(testloader)):
            image = image.cuda()
            image2 = image2.cuda()
            label = label.cuda()

            I, R = Decomp(image)
            I2, R2 = Decomp(image2)
            R_hat, smap = Enhance(R, I)


            # 更新统计量
            metric.update(smap, label)
            cityscapes_classes = [
                "unlabeled", "bed", "books", "ceil", "chair", "floor", "furn.", "obj.", "pai.", "sofa", "table", "tv",
                "wall", "wind"
            ]
            # 计算并打印结果
            results = metric.compute(class_names=cityscapes_classes)
            # 可视化分割图（one-hot -> 颜色图）
            smap_oh = utils.reverse_one_hot(smap)
            sout = smap_oh[0, :, :]  # [H, W]
            sout = utils.colorize(sout).numpy()  # 转为 RGB 可视化图像
            sout = np.transpose(sout, (2, 0, 1))  # [C, H, W]

            # 保存 sout
            os.makedirs('test_result', exist_ok=True)
            output_name = name[0]
            if not output_name.endswith(('.png', '.jpg', '.jpeg')):
                output_name += '.jpg'  # 默认加上 .jpg

            sout_img = np.transpose(sout, (1, 2, 0))  # [H, W, C]

            # 如果是 float 类型，先乘255并转为 uint8
            if sout_img.dtype != np.uint8:
                sout_img = np.clip(sout_img * 255.0, 0, 255).astype(np.uint8)

            # 确保图像尺寸正常，排除极小尺寸误输入
            if sout_img.shape[0] < 10 or sout_img.shape[1] < 10:
                print(f"[警告] 图像尺寸异常: {sout_img.shape}，跳过保存 {name[0]}")
                continue

            # 保存图像
            Image.fromarray(sout_img).save(os.path.join('test_result', f"{os.path.splitext(name[0])[0]}_smap.jpg"))


# def test(args, Decomp, Enhance, testloader):
#     print('Start Testing...')
#     Decomp.eval()
#     Enhance.eval()
#
#     os.makedirs('test_result', exist_ok=True)
#
#     with torch.no_grad():
#         for i, (image, image2, label, name) in enumerate(tqdm(testloader)):
#             image = image.cuda()
#             image2 = image2.cuda()
#
#             # Decomposition
#             I, R = Decomp(image)
#             # image2 仅用于网络推理，但不使用其结果
#             _ = Decomp(image2)
#
#             # Enhancement
#             R_hat, _ = Enhance(R, I)
#
#             # 恢复图像 I_e
#             I_3 = torch.cat((I, I, I), dim=1)
#             I_pow = torch.pow(I_3, 0.1)
#             I_e = I_pow * R_hat
#
#             # 保存 I_e 彩色图像
#             I_e_img = I_e[0].cpu().numpy().transpose(1, 2, 0)
#             I_e_img = np.clip(I_e_img * 255.0, 0, 255.0).astype('uint8')
#
#             basename = os.path.splitext(name[0])[0]
#             Image.fromarray(I_e_img).save(os.path.join('test_result', f'{basename}.jpg'))


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default='weights/model_best.pth.tar', help='path to the best checkpoint')
    parser.add_argument('--data_path', type=str, default='LISU_LLRGBD_real', help='path to your dataset')
    parser.add_argument('--crop_height', type=int, default=320, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=240, help='Width of cropped/resized input image to network')
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--multiple_GPUs', action='store_true')
    args = parser.parse_args()

    # 模型加载
    Decomp = models.DECOMP_net().cuda()
    Enhance = models.Seg_net().cuda()

    if os.path.isfile(args.model_path):
        print(f"=> loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path)
        Decomp.load_state_dict(checkpoint['state_dict_decomp'])
        Enhance.load_state_dict(checkpoint['state_dict_enhance'])
    else:
        print(f"=> no checkpoint found at '{args.model_path}'")
        return

    # 数据加载器
    valset = LLRGBD_real(args, mode='val')
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # 测试
    test(args, Decomp, Enhance, valloader)


if __name__ == '__main__':
    main()
