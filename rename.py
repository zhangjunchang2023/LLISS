import os
import shutil

# 原始目录与目标目录
src_root = './data/val'
dst_root = './data/val2'

# 子文件夹名称
sub_dirs = ['val_H', 'val_L', 'val_M']

# 创建目标目录结构
for sub in sub_dirs:
    os.makedirs(os.path.join(dst_root, sub), exist_ok=True)

# 遍历每个子文件夹并进行重命名复制
for sub in sub_dirs:
    src_path = os.path.join(src_root, sub)
    dst_path = os.path.join(dst_root, sub)

    # 获取并排序文件名
    file_list = sorted(os.listdir(src_path))

    # 遍历重命名并复制
    for idx, filename in enumerate(file_list):
        ext = os.path.splitext(filename)[1]  # 获取扩展名
        new_name = f"{idx + 1}{ext}"         # 构建新文件名
        src_file = os.path.join(src_path, filename)
        dst_file = os.path.join(dst_path, new_name)
        shutil.copy(src_file, dst_file)

print("重命名并保存到 data/val2 完成！")
