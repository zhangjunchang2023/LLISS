import torch
import numpy as np
from collections import OrderedDict
from prettytable import PrettyTable


class IoUMetric:
    """IoU评估工具，支持紧凑表格输出"""

    def __init__(self, num_classes, ignore_index=255, device='cuda'):
        """
        Args:
            num_classes (int): 包含背景的类别总数
            ignore_index (int): 要忽略的标签索引
            device (str): 计算设备
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()

    def reset(self):
        """重置所有统计量"""
        self.intersect = torch.zeros(self.num_classes, device=self.device)
        self.union = torch.zeros(self.num_classes, device=self.device)
        self.pred_area = torch.zeros(self.num_classes, device=self.device)
        self.label_area = torch.zeros(self.num_classes, device=self.device)

    def update(self, preds, labels):
        """
        更新统计量

        Args:
            preds (Tensor): 模型输出 [B, C, H, W]
            labels (Tensor): 真实标签 [B, H, W]
        """
        # 转换预测结果为类别索引
        preds = preds.argmax(dim=1)  # [B, H, W]

        batch_size = preds.shape[0]
        for i in range(batch_size):
            pred = preds[i]  # [H, W]
            label = labels[i]  # [H, W]

            # 计算交集和并集
            intersect, union, area_pred, area_label = self._compute_stats(pred, label)

            # 累积统计量
            self.intersect += intersect
            self.union += union
            self.pred_area += area_pred
            self.label_area += area_label

    def _compute_stats(self, pred, label):
        """计算单个样本的交并统计"""
        mask = (label != self.ignore_index)
        pred = pred[mask]
        label = label[mask]

        # 计算交集
        intersect = pred[pred == label]
        area_intersect = torch.histc(
            intersect.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

        # 计算预测和标签的面积
        area_pred = torch.histc(
            pred.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        area_label = torch.histc(
            label.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

        # 计算并集
        area_union = area_pred + area_label - area_intersect

        return area_intersect, area_union, area_pred, area_label

    def compute(self, class_names=None):
        """计算并返回指标结果"""
        # 转换为CPU numpy数组
        intersect = self.intersect.cpu().numpy()
        union = self.union.cpu().numpy()
        label_area = self.label_area.cpu().numpy()

        # 计算各类IoU
        iou_per_class = np.divide(intersect, union + 1e-10,
                                  out=np.zeros_like(intersect),
                                  where=union != 0)

        # 计算总体指标
        total_intersect = np.sum(intersect)
        total_union = np.sum(union)
        miou = total_intersect / (total_union + 1e-10)

        # 构建结果字典
        results = {
            'mIoU': miou,
            'iou_per_class': iou_per_class
        }

        # 打印表格
        if class_names is not None:
            self._print_compact_table(iou_per_class, class_names)

        return results

    def _print_compact_table(self, iou_per_class, class_names):
        """打印紧凑型5行7列表格"""
        # 表格参数设置
        COLS = 7
        ROWS = 5

        # 创建表格对象
        table = PrettyTable()
        table.field_names = [f"Class/IoU {i + 1}" for i in range(COLS)]
        table.align = "l"
        table.horizontal_char = "="
        table.junction_char = "="

        # 填充数据
        for row in range(ROWS):
            row_data = []
            for col in range(COLS):
                idx = row * COLS + col
                if idx < len(class_names):
                    name = class_names[idx][:12]  # 限制名称长度
                    iou = iou_per_class[idx]
                    cell = f"{name}\n{iou:.4f}"
                else:
                    cell = ""
                row_data.append(cell)

            # 添加行（跳过全空的行）
            if any(row_data):
                table.add_row(row_data)

        # 添加mIoU行（合并单元格）
        mIoU_row = [f"mIoU: {np.nanmean(iou_per_class):.4f}"] + [""] * (COLS - 1)
        table.add_row(mIoU_row)

        # 打印表格
        print("\n" + "=" * 30 + " Evaluation Results " + "=" * 30)
        print(table)
        print("=" * 80 + "\n")


# 使用示例（Cityscapes 34类）
if __name__ == "__main__":
    # Cityscapes类别名称（示例）
    cityscapes_classes = [
        "unlabeled", "bed","books","ceil","chair","floor","furn.","obj.","pai.","sofa","table","tv","wall","wind"
,
    ]

    # 初始化评估器
    metric = IoUMetric(num_classes=14, device='cpu')

    # 模拟验证数据
    batch_size = 4
    preds = torch.randn(batch_size, 34, 512, 512)
    labels = torch.randint(0, 34, (batch_size, 512, 512))

    # 更新统计量
    metric.update(preds, labels)

    # 计算并打印结果
    results = metric.compute(class_names=cityscapes_classes)

# import torch
# import numpy as np
# from collections import OrderedDict
# from prettytable import PrettyTable
#
#
# class IoUMetric:
#     """IoU评估工具，支持类别级指标输出"""
#
#     def __init__(self, num_classes, ignore_index=255, device='cuda'):
#         """
#         Args:
#             num_classes (int): 包含背景的类别总数
#             ignore_index (int): 要忽略的标签索引
#             device (str): 计算设备
#         """
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
#         self.device = device
#         self.reset()
#
#     def reset(self):
#         """重置所有统计量"""
#         self.intersect = torch.zeros(self.num_classes, device=self.device)
#         self.union = torch.zeros(self.num_classes, device=self.device)
#         self.pred_area = torch.zeros(self.num_classes, device=self.device)
#         self.label_area = torch.zeros(self.num_classes, device=self.device)
#
#     def update(self, preds, labels):
#         """
#         更新统计量
#
#         Args:
#             preds (Tensor): 模型输出 [B, C, H, W]
#             labels (Tensor): 真实标签 [B, H, W]
#         """
#         # 转换预测结果为类别索引
#         preds = preds.argmax(dim=1)  # [B, H, W]
#
#         batch_size = preds.shape[0]
#         for i in range(batch_size):
#             pred = preds[i]  # [H, W]
#             label = labels[i]  # [H, W]
#
#             # 计算交集和并集
#             intersect, union, area_pred, area_label = self._compute_stats(pred, label)
#
#             # 累积统计量
#             self.intersect += intersect
#             self.union += union
#             self.pred_area += area_pred
#             self.label_area += area_label
#
#     def _compute_stats(self, pred, label):
#         """计算单个样本的交并统计"""
#         mask = (label != self.ignore_index)
#         pred = pred[mask]
#         label = label[mask]
#
#         # 计算交集
#         intersect = pred[pred == label]
#         area_intersect = torch.histc(
#             intersect.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
#
#         # 计算预测和标签的面积
#         area_pred = torch.histc(
#             pred.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
#         area_label = torch.histc(
#             label.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
#
#         # 计算并集
#         area_union = area_pred + area_label - area_intersect
#
#         return area_intersect, area_union, area_pred, area_label
#
#     def compute(self, class_names=None):
#         """计算并返回指标结果"""
#         # 转换为CPU numpy数组
#         intersect = self.intersect.cpu().numpy()
#         union = self.union.cpu().numpy()
#         label_area = self.label_area.cpu().numpy()
#
#         # 计算各类IoU
#         iou_per_class = np.divide(intersect, union + 1e-10,
#                                   out=np.zeros_like(intersect),
#                                   where=union != 0)
#
#         # 计算总体指标
#         total_intersect = np.sum(intersect)
#         total_union = np.sum(union)
#         miou = total_intersect / (total_union + 1e-10)
#
#         # 构建结果字典
#         results = {
#             'mIoU': miou,
#             'iou_per_class': iou_per_class
#         }
#
#         # 打印表格
#         if class_names is not None:
#             self._print_table(iou_per_class, class_names)
#
#         return results
#
#     def _print_table(self, iou_per_class, class_names):
#         """打印类别级IoU表格"""
#         table = PrettyTable()
#         table.field_names = ["Class", "IoU"]
#         table.float_format = ".3"
#
#         for idx, (name, iou) in enumerate(zip(class_names, iou_per_class)):
#             table.add_row([f"{idx} {name}", f"{iou:.4f}"])
#
#         # 添加均值行
#         table.add_row(["mIoU", f"{np.nanmean(iou_per_class):.4f}"])
#
#         print("\n" + "=" * 30 + " Evaluation Results " + "=" * 30)
#         print(table)
#         print("=" * 80 + "\n")
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 参数设置
#     num_classes = 5
#     class_names = ["background", "cat", "dog", "car", "person"]
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # 初始化评估器
#     metric = IoUMetric(num_classes=num_classes, device=device)
#
#     # 模拟验证数据
#     batch_size = 4
#     preds = torch.randn(batch_size, num_classes, 256, 256).to(device)  # 模型输出logits
#     labels = torch.randint(0, num_classes, (batch_size, 256, 256)).to(device)
#
#     # 更新统计量
#     metric.update(preds, labels)
#
#     # 计算并打印结果
#     results = metric.compute(class_names=class_names)
#     print(f"Overall mIoU: {results['mIoU']:.4f}")
# import torch
# import numpy as np
# from collections import OrderedDict
# from prettytable import PrettyTable
#
#
#
# class IoUMetric:
#     """IoU 评估模块 (支持多卡分布式训练)"""
#
#     def __init__(self, num_classes, ignore_index=255, metrics=['mIoU'], beta=1, nan_to_num=None):
#         """
#         参数:
#             num_classes (int): 类别数量（含背景）
#             ignore_index (int): 要忽略的标签索引，默认255
#             metrics (list): 评估指标，可选'mIoU', 'mDice', 'mFscore'
#             beta (int): F-score计算中的beta值
#             nan_to_num (float): 替换NaN值的数值
#         """
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
#         self.metrics = metrics
#         self.beta = beta
#         self.nan_to_num = nan_to_num
#
#         # 初始化统计量
#         self.reset()
#
#     def reset(self):
#         """重置累积统计量"""
#         self.total_intersect = torch.zeros(self.num_classes)
#         self.total_union = torch.zeros(self.num_classes)
#         self.total_pred = torch.zeros(self.num_classes)
#         self.total_label = torch.zeros(self.num_classes)
#
#     def update(self, preds, labels):
#         """
#         更新统计量
#
#         参数:
#             preds (Tensor): [B, H, W] 预测的分割图 (类别索引)
#             labels (Tensor): [B, H, W] 真实标签
#         """
#         # 确保设备一致
#         preds = preds.to(labels.device)
#
#         for pred, label in zip(preds, labels):
#             intersect, union, pred_area, label_area = self.intersect_and_union(
#                 pred, label, self.num_classes, self.ignore_index)
#
#             # 累积统计量
#             self.total_intersect += intersect
#             self.total_union += union
#             self.total_pred += pred_area
#             self.total_label += label_area
#
#     def compute(self, class_names=None):
#         """计算最终指标"""
#         # 转换统计量为numpy
#         intersect = self.total_intersect.numpy()
#         union = self.total_union.numpy()
#         pred_area = self.total_pred.numpy()
#         label_area = self.total_label.numpy()
#
#         # 计算各类指标
#         metrics_dict = self.calculate_metrics(intersect, union, pred_area, label_area)
#
#         # 打印表格
#         if class_names is not None and len(class_names) == self.num_classes:
#             self.print_class_table(metrics_dict, class_names)
#
#         return metrics_dict
#
#     def intersect_and_union(self, pred, label, num_classes, ignore_index):
#         """计算单个样本的交并统计"""
#         mask = (label != ignore_index)
#         pred = pred[mask]
#         label = label[mask]
#
#         # 计算直方图
#         intersect = pred[pred == label]
#         area_intersect = torch.histc(intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
#         area_pred = torch.histc(pred.float(), bins=num_classes, min=0, max=num_classes - 1)
#         area_label = torch.histc(label.float(), bins=num_classes, min=0, max=num_classes - 1)
#         area_union = area_pred + area_label - area_intersect
#
#         return area_intersect, area_union, area_pred, area_label
#
#     def calculate_metrics(self, intersect, union, pred, label):
#         """核心指标计算"""
#         metrics = OrderedDict()
#
#         # 整体准确率
#         total_acc = np.nansum(intersect) / np.nansum(label)
#         metrics['aAcc'] = total_acc
#
#         # 各类别指标
#         iou = intersect / (union + 1e-10)
#         dice = 2 * intersect / (pred + label + 1e-10)
#
#         if 'mIoU' in self.metrics:
#             metrics['mIoU'] = np.nanmean(iou)
#         if 'mDice' in self.metrics:
#             metrics['mDice'] = np.nanmean(dice)
#         if 'mFscore' in self.metrics:
#             precision = intersect / (pred + 1e-10)
#             recall = intersect / (label + 1e-10)
#             f_score = (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall + 1e-10)
#             metrics['mFscore'] = np.nanmean(f_score)
#
#         # 处理NaN值
#         if self.nan_to_num is not None:
#             for k, v in metrics.items():
#                 metrics[k] = np.nan_to_num(v, nan=self.nan_to_num)
#
#         return metrics
#
#     def print_class_table(self, metrics_dict, class_names):
#         """打印类别级表格"""
#         table = PrettyTable()
#         table.field_names = ["Class", "IoU", "Dice"]
#         for i, name in enumerate(class_names):
#             iou = metrics_dict.get('IoU', [0] * self.num_classes)[i]
#             dice = metrics_dict.get('Dice', [0] * self.num_classes)[i]
#             table.add_row([name, f"{iou:.2f}", f"{dice:.2f}"])
#         print("\nPer-Class Results:")
#         print(table)
#
# if __name__ == '__main__':
#     # 初始化评估器
#     num_classes = 19
#     class_names = [f"class_{i}" for i in range(num_classes)]
#     metric = IoUMetric(num_classes=num_classes, metrics=['mIoU', 'mDice'])
#
#     # 模拟数据
#     batch_size = 4
#     preds = torch.randint(0, num_classes, (batch_size, 512, 512))  # 预测结果
#     labels = torch.randint(0, num_classes, (batch_size, 512, 512))  # 真实标签
#
#     # 更新统计量
#     metric.update(preds, labels)
#
#     # 计算指标
#     results = metric.compute(class_names=class_names)
#     print(f"mIoU: {results['mIoU']:.2f}% | mDice: {results['mDice']:.2f}%")