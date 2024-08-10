import os
import cv2
import numpy as np
from PIL import Image

def get_otsu(img):
    """
    对图像应用Otsu二值化。

    参数:
    img (PIL.Image): 输入图像

    返回:
    PIL.Image: Otsu二值化后的图像
    """
    img_gray = img.convert('L')  # 转换为灰度图
    img_gray_np = np.asarray(img_gray).astype(np.uint8)  # 转换为numpy数组
    _, th2 = cv2.threshold(img_gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 应用Otsu二值化
    return Image.fromarray(th2)  # 转换回PIL图像

# 文件夹路径
img_folder = "../results/A4/c16"
otsu_folder = "../results/A4/c16"

# 创建otsu文件夹，如果不存在
os.makedirs(otsu_folder, exist_ok=True)

# 遍历img文件夹中的所有文件
for filename in os.listdir(img_folder):
    img_path = os.path.join(img_folder, filename)  # 完整图像路径
    img = Image.open(img_path)  # 打开图像

    # 获取Otsu二值化图像
    otsu_img = get_otsu(img)

    # 保存Otsu二值化图像
    otsu_save_path = os.path.join(otsu_folder, filename)  # 生成保存路径
    otsu_img.save(otsu_save_path)  # 保存图像

    print(f"Otsu image saved: {otsu_save_path}")
