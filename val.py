
import os
import shutil

import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
import matplotlib
from tensorboardX import SummaryWriter

matplotlib.use('Agg')
from loss import *
from dataset import LLRGBD_real, LLRGBD_synthetic
from PIL import Image
import utils

import models
import argparse
import sys
import time

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile='full')
torch.backends.cudnn.benchmark = True








def test(args, Decomp, Enhance, testloader):
    print('Start Testing...')
    Decomp.eval()
    Enhance.eval()

    with torch.no_grad():
        for i, (image, image2, label, name) in enumerate(tqdm(testloader)):
            image = image.cuda()
            image2 = image2.cuda()

            I, R = Decomp(image)
            I2, R2 = Decomp(image2)
            R_hat, smap = Enhance(R, I)

            I_3 = torch.cat((I, I, I), dim=1)
            I_pow = torch.pow(I_3, 0.1)
            I_e = I_pow * R_hat

            smap_oh = utils.reverse_one_hot(smap)
            sout = smap_oh[0, :, :]
            sout = utils.colorize(sout).numpy()
            sout = np.transpose(sout, (2, 0, 1))

            # 单通道 I, I2 可视化处理（转为灰度图）
            I_img = I[0, 0, :, :].cpu().numpy()
            I2_img = I2[0, 0, :, :].cpu().numpy()

            # 三通道图像：直接取 tensor
            L_img = image[0].cpu().numpy().transpose(1, 2, 0)
            H_img = image2[0].cpu().numpy().transpose(1, 2, 0)
            R_img = R[0].cpu().numpy().transpose(1, 2, 0)
            R2_img = R2[0].cpu().numpy().transpose(1, 2, 0)

            # 归一化到 [0,255]
            L_img = np.clip(L_img * 255, 0, 255).astype(np.uint8)
            H_img = np.clip(H_img * 255, 0, 255).astype(np.uint8)
            R_img = np.clip(R_img * 255, 0, 255).astype(np.uint8)
            R2_img = np.clip(R2_img * 255, 0, 255).astype(np.uint8)
            I_img = np.clip(I_img * 255, 0, 255).astype(np.uint8)
            I2_img = np.clip(I2_img * 255, 0, 255).astype(np.uint8)

            # 保存图像
            os.makedirs('test_result', exist_ok=True)
            basename = os.path.splitext(name[0])[0]

            Image.fromarray(L_img).save(os.path.join('test_result', f'{basename}_L_image.jpg'))
            Image.fromarray(I_img).save(os.path.join('test_result', f'{basename}_I_L.jpg'))
            Image.fromarray(R_img).save(os.path.join('test_result', f'{basename}_R_L.jpg'))
            Image.fromarray(H_img).save(os.path.join('test_result', f'{basename}_H_image.jpg'))
            Image.fromarray(I2_img).save(os.path.join('test_result', f'{basename}_I_H.jpg'))
            Image.fromarray(R2_img).save(os.path.join('test_result', f'{basename}_R_H.jpg'))

            # 拼接图像可视化
            cat_image = np.concatenate(
                [image[0, :, :, :].cpu(), R[0, :, :, :].cpu(), R_hat[0, :, :, :].cpu(), I_e[0, :, :, :].cpu(), sout], axis=2)
            cat_image = np.transpose(cat_image, (1, 2, 0))
            cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')



            # 保存结果
            os.makedirs('test_result', exist_ok=True)
            output_name = name[0]
            if not output_name.endswith(('.png', '.jpg', '.jpeg')):
                output_name += '.jpg'  # 默认加上 .jpg

            Image.fromarray(cat_image).save(os.path.join('test_result', output_name))





def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default='weights/model_best.pth.tar', help='path to the best checkpoint')
    parser.add_argument('--data_path', type=str, default='LISU_LLRGBD_real', help='path to your dataset')
    parser.add_argument('--crop_height', type=int, default=320, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=240, help='Width of cropped/resized input image to network')
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--multiple_GPUs', action='store_true')
    args = parser.parse_args()

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
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)



    test(args, Decomp, Enhance, valloader)

if __name__ == '__main__':
    main()
