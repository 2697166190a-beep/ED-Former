import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# --- 1. 字体配置：使用与 Figure 8 一致的学术无衬线字体 (Arial) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- 配置区域 ---
folder1_path = '1'
folder2_path = '2'

labels = ["Input", "Ours", "GT"]

# --- 代码实现 ---

if not os.path.isdir(folder1_path) or not os.path.isdir(folder2_path):
    print(f"错误：请确保文件夹 '{folder1_path}' 和 '{folder2_path}' 存在。")
    sys.exit()

try:
    images_set1 = sorted([os.path.join(folder1_path, f) for f in os.listdir(folder1_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))])
    images_set2 = sorted([os.path.join(folder2_path, f) for f in os.listdir(folder2_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))])
except Exception as e:
    print(f"读取图片文件时出错: {e}")
    sys.exit()

if len(images_set1) < 3 or len(images_set2) < 3:
    print(f"错误：请确保每个文件夹中至少包含 3 张图片。")
    sys.exit()

# 【完美尺寸】宽度 8.0，高度 4.8（左右不宽，上下不空）
fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.8))
ax = axes.ravel()

# --- 绘制第一组图片 (第一行) ---
for i in range(3):
    try:
        img = Image.open(images_set1[i])
        ax[i].imshow(img)
        # y=-0.15 保持贴近，字号 11，不加粗（标准 Arial 字体）
        ax[i].set_title(labels[i], y=-0.15, fontsize=11) 
    except Exception as e:
        ax[i].set_title(f"{labels[i]}\n(加载失败)", y=-0.15, fontsize=10)
    ax[i].axis('off')

# --- 绘制第二组图片 (第二行) ---
for i in range(3):
    ax_index = i + 3
    try:
        img = Image.open(images_set2[i])
        ax[ax_index].imshow(img)
        ax[ax_index].set_title(labels[i], y=-0.15, fontsize=11) 
    except Exception as e:
        ax[ax_index].set_title(f"{labels[i]}\n(加载失败)", y=-0.15, fontsize=10)
    ax[ax_index].axis('off')

# 使用 tight_layout 并保持极致紧凑的间距（wspace=0.01，hspace=0.05）
plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.05)

output_filename = "failure_case_comparison.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"图像已成功保存为 '{output_filename}'")

plt.show()
