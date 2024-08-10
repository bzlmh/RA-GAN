import cv2
import os


# def simple_threshold(image):
#     # 将图像转为灰度图
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 应用固定阈值二值化
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
#     return binary


# 输入文件夹和输出文件夹路径
input_folder = '../datasets/HDIBCO/img'
output_folder = '../datasets/img'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的图像文
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith(
            '.tif') or filename.endswith('.tiff') or filename.endswith('.bmp'):
        # 读取图像
        image = cv2.imread(os.path.join(input_folder, filename))
        #
        # # 调用普通二值化算法
        # binary_image = simple_threshold(image)

        # 构建输出文件路径
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

        # 保存二值化结果
        cv2.imwrite(output_path,image)

        print(f'{filename} 处理完成')

print('所有图像处理完成')
