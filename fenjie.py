# # import torch
# # import torch.nn as nn
# # import torchvision.transforms as transforms
# # from PIL import Image
# # import numpy as np
# # import os
# #
# #
# # # ----------------------------
# # # 网络定义
# # # ----------------------------
# # class Illumination_Estimator(nn.Module):
# #     def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
# #         super(Illumination_Estimator, self).__init__()
# #         self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
# #         self.depth_conv = nn.Conv2d(
# #             n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
# #         self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
# #
# #     def forward(self, img, mean_c):
# #         input = torch.cat([img, mean_c], dim=1)  # [B,4,H,W]
# #         x_1 = self.conv1(input)
# #         illu_fea = self.depth_conv(x_1)
# #         illu_map = self.conv2(illu_fea)
# #         return illu_fea, illu_map
# #
# #
# # # ----------------------------
# # # 加载图片工具函数
# # # ----------------------------
# # def load_rgb_image(path, size=(256, 256)):
# #     transform = transforms.Compose([
# #         transforms.Resize(size),
# #         transforms.ToTensor(),  # 输出 [3,H,W], 范围[0,1]
# #     ])
# #     img = Image.open(path).convert('RGB')
# #     return transform(img).unsqueeze(0)  # [1,3,H,W]
# #
# #
# # def load_gray_image(path, size=(256, 256)):
# #     transform = transforms.Compose([
# #         transforms.Resize(size),
# #         transforms.ToTensor(),  # 输出 [1,H,W], 范围[0,1]
# #     ])
# #     img = Image.open(path).convert('L')
# #     return transform(img).unsqueeze(0)  # [1,1,H,W]
# #
# #
# # # ----------------------------
# # # 主函数
# # # ----------------------------
# # def main():
# #     model = Illumination_Estimator(n_fea_middle=32).cuda()
# #     model.eval()
# #
# #     # 替换为你自己的图像路径
# #     path_R = './illu_output/input/227R.jpg'     # 三通道反射率图
# #     path_I = './illu_output/input/227I.jpg'     # 单通道照度图
# #
# #     R = load_rgb_image(path_R).cuda()  # [1,3,H,W]
# #     I = load_gray_image(path_I).cuda()  # [1,1,H,W]
# #
# #     with torch.no_grad():
# #         illu_fea, illu_map = model(R, I)
# #
# #     os.makedirs('illu_output', exist_ok=True)
# #
# #     # 保存 illu_map（RGB）
# #     illu_map_img = illu_map[0].permute(1, 2, 0).cpu().numpy()  # [H,W,3]
# #     illu_map_img = np.clip(illu_map_img * 255, 0, 255).astype(np.uint8)
# #     Image.fromarray(illu_map_img).save('illu_output/illu_map_rgb.jpg')
# #
# #     # 保存灰度图（通道平均）
# #     illu_map_gray = torch.mean(illu_map[0], dim=0).cpu().numpy()  # [H,W]
# #     illu_map_gray = np.clip(illu_map_gray * 255, 0, 255).astype(np.uint8)
# #     Image.fromarray(illu_map_gray).save('illu_output/illu_map_gray.jpg')
# #
# #     # 保存特征图平均图
# #     illu_fea_avg = torch.mean(illu_fea[0], dim=0).cpu().numpy()
# #     illu_fea_avg = np.clip(illu_fea_avg * 255, 0, 255).astype(np.uint8)
# #     Image.fromarray(illu_fea_avg).save('illu_output/illu_fea_avg.jpg')
# #
# #     print("✓ 推理完成，结果保存在 'illu_output' 文件夹中")
# #
# #
# # if __name__ == '__main__':
# #     main()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import os
# import cv2
#
#
# # ----------------------------
# # 网络定义
# # ----------------------------
# class Illumination_Estimator(nn.Module):
#     def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
#         super(Illumination_Estimator, self).__init__()
#         self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
#         self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
#         self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
#
#     def forward(self, img, mean_c):
#         input = torch.cat([img, mean_c], dim=1)  # [B,4,H,W]
#         x_1 = self.conv1(input)
#         illu_fea = self.depth_conv(x_1)
#         illu_map = self.conv2(illu_fea)
#         return illu_fea, illu_map
#
#
# # ----------------------------
# # 加载图片工具函数
# # ----------------------------
# def load_rgb_image(path, size=(256, 256)):
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor(),
#     ])
#     img = Image.open(path).convert('RGB')
#     return transform(img).unsqueeze(0)  # [1,3,H,W]
#
#
# def load_gray_image(path, size=(256, 256)):
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor(),
#     ])
#     img = Image.open(path).convert('L')
#     return transform(img).unsqueeze(0)  # [1,1,H,W]
#
#
# # ----------------------------
# # 二值化+双色上色
# # ----------------------------
# def binary_color_map(gray_img, threshold=50, high_color=(0, 0, 0), low_color=(0, 0, 32 )):
#     """
#     二值图上色：大于threshold为 high_color，小于等于为 low_color
#     gray_img: numpy [H,W], 范围 0~255
#     返回值: 彩色图 [H,W,3]
#     """
#     h, w = gray_img.shape
#     color_img = np.zeros((h, w, 3), dtype=np.uint8)
#
#     high_mask = gray_img > threshold
#     low_mask = ~high_mask
#
#     color_img[high_mask] = high_color
#     color_img[low_mask] = low_color
#
#     return color_img
#
#
# # ----------------------------
# # 主函数
# # ----------------------------
# def main():
#     model = Illumination_Estimator(n_fea_middle=32).cuda()
#     model.eval()
#
#     # 图像路径
#     path_R = './illu_output/input/227R.jpg'  # 三通道反射率图
#     path_I = './illu_output/input/227I.jpg'  # 单通道照度图
#
#     # 加载图像
#     R = load_rgb_image(path_R).cuda()  # [1,3,H,W]
#     I = load_gray_image(path_I).cuda()  # [1,1,H,W]
#
#     with torch.no_grad():
#         illu_fea, illu_map = model(R, I)
#
#     os.makedirs('illu_output', exist_ok=True)
#
#     # 保存照度图 RGB
#     illu_map_img = illu_map[0].permute(1, 2, 0).cpu().numpy()  # [H,W,3]
#     illu_map_img = np.clip(illu_map_img * 255, 0, 255).astype(np.uint8)
#     Image.fromarray(illu_map_img).save('illu_output/illu_map_rgb.jpg')
#
#     # 保存照度图灰度图（通道平均）
#     # illu_map_gray = torch.mean(illu_map[0], dim=0).cpu().numpy()  # [H,W]
#     # illu_map_gray = np.clip(illu_map_gray * 255, 0, 255).astype(np.uint8)
#     # Image.fromarray(illu_map_gray).save('illu_output/illu_map_gray.jpg')
#     # 先将 numpy 数组转换为 Pillow 图像对象
#     illu_map_pil = Image.fromarray(illu_map_img)  # RGB 图
#     # 转换为灰度图（L 模式）
#     illu_map_gray = illu_map_pil.convert('L')  # 单通道灰度图
#     # 保存灰度图
#     illu_map_gray.save('illu_output/illu_map_gray.jpg')
#     illu_map_gray =  np.array(illu_map_gray)
#
#     # 保存照度图上色（红/蓝）
#     illu_map_binary_color = binary_color_map(illu_map_gray,threshold=2)
#     cv2.imwrite('illu_output/illu_map_binary_color.jpg', cv2.cvtColor(illu_map_binary_color, cv2.COLOR_RGB2BGR))
#
#     # 保存特征图平均图
#     illu_fea_avg = torch.mean(illu_fea[0], dim=0).cpu().numpy()
#     illu_fea_avg = np.clip(illu_fea_avg * 255, 0, 255).astype(np.uint8)
#     Image.fromarray(illu_fea_avg).save('illu_output/illu_fea_avg.jpg')
#
#     # 保存特征图上色（红/蓝）
#     illu_fea_binary_color = binary_color_map(illu_fea_avg)
#     cv2.imwrite('illu_output/illu_fea_avg_binary_color.jpg', cv2.cvtColor(illu_fea_binary_color, cv2.COLOR_RGB2BGR))
#
#     print("✓ 推理完成，结果保存在 'illu_output' 文件夹中")
#
#
# if __name__ == '__main__':
#     main()

















#
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
#
# class Illumination_Estimator(nn.Module):
#     def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
#         super(Illumination_Estimator, self).__init__()
#         self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
#         self.depth_conv = nn.Conv2d(
#             n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
#         self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
#
#     def forward(self, img, mean_c):
#         input = torch.cat([img, mean_c], dim=1)  # [B,4,H,W]
#         x_1 = self.conv1(input)
#         illu_fea = self.depth_conv(x_1)
#         illu_map = self.conv2(illu_fea)
#         return illu_fea, illu_map
#
#
# def load_rgb_image(path, size=(256, 256)):
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor(),
#     ])
#     img = Image.open(path).convert('RGB')
#     return transform(img).unsqueeze(0)
#
#
# def load_gray_image(path, size=(256, 256)):
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor(),
#     ])
#     img = Image.open(path).convert('L')
#     return transform(img).unsqueeze(0)
#
#
# def normalize_to_u8(x):
#     x = x - x.min()
#     x = x / (x.max() + 1e-6)
#     return (x * 255).astype(np.uint8)
#
#
# def apply_colormap(gray_img_u8, cmap_name='cividis'):
#     cmap = plt.get_cmap(cmap_name)
#     color_img = cmap(gray_img_u8 / 255.0)[..., :3]
#     color_img_u8 = (color_img * 255).astype(np.uint8)
#     return Image.fromarray(color_img_u8)
#
#
# def main():
#     model = Illumination_Estimator(n_fea_middle=32).cuda()
#     model.eval()
#
#     path_R = './illu_output/input/227R.jpg'
#     path_I = './illu_output/input/227I.jpg'
#
#     R = load_rgb_image(path_R).cuda()
#     I = load_gray_image(path_I).cuda()
#
#     with torch.no_grad():
#         illu_fea, illu_map = model(R, I)
#
#     os.makedirs('illu_output', exist_ok=True)
#
#     # 保存 illu_map RGB
#     illu_map_img = illu_map[0].permute(1, 2, 0).cpu().numpy()
#     illu_map_img = np.clip(illu_map_img * 255, 0, 255).astype(np.uint8)
#     Image.fromarray(illu_map_img).save('illu_output/illu_map_rgb.jpg')
#
#     # 灰度图
#     illu_map_gray = torch.mean(illu_map[0], dim=0).cpu().numpy()
#     illu_map_gray_u8 = normalize_to_u8(illu_map_gray)
#     Image.fromarray(illu_map_gray_u8).save('illu_output/illu_map_gray.jpg')
#
#     # 彩色伪彩图（cividis）
#     color_map_img = apply_colormap(illu_map_gray_u8, cmap_name='cividis')
#     color_map_img.save('illu_output/illu_map_color.jpg')
#
#     # 特征图平均 + 增强处理
#     illu_fea_avg = torch.mean(illu_fea[0], dim=0).cpu().numpy()
#     illu_fea_avg_u8 = normalize_to_u8(illu_fea_avg)
#     Image.fromarray(illu_fea_avg_u8).save('illu_output/illu_fea_avg_gray.jpg')
#
#     color_fea_img = apply_colormap(illu_fea_avg_u8, cmap_name='turbo')  # 更明显
#     color_fea_img.save('illu_output/illu_fea_avg_color.jpg')
#
#     print("✓ 完成！输出图像包含 RGB、灰度图、彩色图和增强特征图")
#
#
# if __name__ == '__main__':
#     main()


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2


# ----------------------------
# 网络定义
# ----------------------------
class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img, mean_c):
        input = torch.cat([img, mean_c], dim=1)  # [B,4,H,W]
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


# ----------------------------
# 加载图片工具函数
# ----------------------------
def load_rgb_image(path, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # 输出 [3,H,W], 范围[0,1]
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)  # [1,3,H,W]


def load_gray_image(path, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # 输出 [1,H,W], 范围[0,1]
    ])
    img = Image.open(path).convert('L')
    return transform(img).unsqueeze(0)  # [1,1,H,W]


# ----------------------------
# 图像增强工具函数
# ----------------------------
def enhance_image(gray_u8):
    # Step 1: 先做双边滤波（保边缘去噪）
    denoised = cv2.bilateralFilter(gray_u8, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Gamma 校正
    gamma = 0.8
    gamma_corrected = np.power(denoised / 255.0, gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    # Step 3: 转换为灰度图（如果是RGB图像）
    if len(gamma_corrected.shape) == 3:  # RGB 图像
        gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2GRAY)
    else:  # 如果已经是灰度图，直接使用
        gray_image = gamma_corrected

    # Step 4: CLAHE（轻度）
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_image)

    # Step 5: 中值滤波（可选，进一步去噪）
    smoothed = cv2.medianBlur(clahe_img, ksize=3)

    return smoothed


# ----------------------------
# 主函数
# ----------------------------
def main():
    model = Illumination_Estimator(n_fea_middle=32).cuda()
    model.eval()

    # 替换为你自己的图像路径
    path_R = './illu_output/input/227R.jpg'  # 三通道反射率图
    path_I = './illu_output/input/227I.jpg'  # 单通道照度图

    R = load_rgb_image(path_R).cuda()  # [1,3,H,W]
    I = load_gray_image(path_I).cuda()  # [1,1,H,W]

    with torch.no_grad():
        illu_fea, illu_map = model(R, I)

    os.makedirs('illu_output', exist_ok=True)

    # 保存 illu_map（RGB）
    illu_map_img = illu_map[0].permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    illu_map_img = np.clip(illu_map_img * 255, 0, 255).astype(np.uint8)

    # 增强图像（锐化+Gamma校正+CLAHE）
    enhanced_illu_map = enhance_image(illu_map_img)

    # 保存增强后的图像
    Image.fromarray(enhanced_illu_map).save('illu_output/illu_map_enhanced.jpg')

    # 保存灰度图（通道平均）
    illu_map_gray = torch.mean(illu_map[0], dim=0).cpu().numpy()  # [H,W]
    illu_map_gray = np.clip(illu_map_gray * 255, 0, 255).astype(np.uint8)
    enhanced_illu_map_gray = enhance_image(illu_map_gray)
    Image.fromarray(enhanced_illu_map_gray).save('illu_output/illu_map_gray_enhanced.jpg')

    # 保存特征图平均图
    illu_fea_avg = torch.mean(illu_fea[0], dim=0).cpu().numpy()
    illu_fea_avg = np.clip(illu_fea_avg * 255, 0, 255).astype(np.uint8)
    enhanced_illu_fea_avg = enhance_image(illu_fea_avg)
    Image.fromarray(enhanced_illu_fea_avg).save('illu_output/illu_fea_avg_enhanced.jpg')

    print("✓ 推理完成，结果保存在 'illu_output' 文件夹中")


if __name__ == '__main__':
    main()
