import os
import cv2
import numpy as np

def pad_image(image, target_height, target_width):
    """
    使用填充对图像进行调整，使其达到目标尺寸。
    """
    height, width = image.shape[:2]

    # 使用白色填充
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # 将原始图像放置在左上角
    padded_image[:height, :width] = image

    return padded_image

def crop_images(input_folders, output_folders, block_size=(256, 256)):
    """
    将图像切割成块，并保存块信息和原始图像大小。
    """
    block_info = {}  # 保存每个图像的块信息
    original_sizes = {}  # 保存每个图像的原始大小

    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            original_sizes[image_file] = (height, width)

            rows = (height + block_size[1] - 1) // block_size[1]
            cols = (width + block_size[0] - 1) // block_size[0]

            block_info[image_file] = []

            for i in range(rows):
                for j in range(cols):
                    x_start = j * block_size[0]
                    y_start = i * block_size[1]
                    x_end = min(x_start + block_size[0], width)
                    y_end = min(y_start + block_size[1], height)

                    block = image[y_start:y_end, x_start:x_end]

                    if block.shape[0] < block_size[1] or block.shape[1] < block_size[0]:
                        block = pad_image(block, block_size[1], block_size[0])

                    output_filename = f"{os.path.splitext(image_file)[0]}_{i}_{j}.png"
                    output_path = os.path.join(output_folder, output_filename)

                    cv2.imwrite(output_path, block)
                    block_info[image_file].append((i, j, y_end - y_start, x_end - x_start))

                    print(f"Saved {output_filename}")

    # 保存块信息和原始大小信息
    np.save("HDIBCO/block_info.npy", block_info)
    np.save("HDIBCO/original_sizes.npy", original_sizes)
    print("Block and original size information saved")

    return block_info, original_sizes
def reconstruct_image(image_file, block_info, original_size, block_folder, block_size=(256, 256)):
    """
    根据块信息重新拼接图像并裁剪到原始大小
    """
    rows = max(block_info, key=lambda x: x[0])[0] + 1
    cols = max(block_info, key=lambda x: x[1])[1] + 1

    reconstructed_image = np.ones((rows * block_size[1], cols * block_size[0], 3), dtype=np.uint8) * 255

    for i, j, orig_height, orig_width in block_info:
        block_filename = f"{os.path.splitext(image_file)[0]}_{i}_{j}.png"
        block_path = os.path.join(block_folder, block_filename)
        block = cv2.imread(block_path)

        y_start = i * block_size[1]
        x_start = j * block_size[0]

        reconstructed_image[y_start:y_start + orig_height, x_start:x_start + orig_width] = block[:orig_height, :orig_width]

    # 根据原始大小裁剪
    original_height, original_width = original_size
    reconstructed_image = reconstructed_image[:original_height, :original_width]

    return reconstructed_image

def save_reconstructed_image(image_file, output_folder, block_folder, block_info, original_size, block_size=(256, 256)):
    """
    保存重新拼接的图像
    """
    reconstructed_image = reconstruct_image(image_file, block_info, original_size, block_folder, block_size)
    output_filename = f"{image_file}"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, reconstructed_image)
    print(f"Saved reconstructed image {output_filename}")
# def crop():
#     # 指定输入和输出文件夹列表
#     input_folders = ["../datasets/HDIBCO/gt"]
#     output_folders = ["../datasets/HDIBCO/ff"]
#     #切割图像并保存块
#     crop_images(input_folders, output_folders)
#     #
#     # # 重新拼接图像
#     # block_folder = "../datasets/dibco/full/img"
#     # output_folder = "../datasets/dibco/full/origin"
#     # if not os.path.exists(output_folder):
#     #     os.makedirs(output_folder)
#     #
#     # # 重新拼接每个图像并保存
#     # for image_file, info in block_info.items():
#     #     save_reconstructed_image(image_file, output_folder, block_folder, info, original_sizes[image_file])

#
def reconstruct_main():
    # 指定块文件夹和输出文件夹
    block_folder = "../results/c16"      # 初始块图像文件夹
    # block_folder2 = "../results/mask_out/"      # 第二个块图像文件夹
    output_folder = "../results/A4/c16"  # 拼接的初始图像输出文件夹
    # output_folder2 = "../datasets/test/mask"    # 拼接的第二个图像输出文件夹

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    # os.makedirs(output_folder2, exist_ok=True)

    # 加载块信息和原始大小信息
    block_info = np.load("HDIBCO/block_infonpy", allow_pickle=True).item()
    original_sizes = np.load("HDIBCO/original_sizes.npy", allow_pickle=True).item()

    # 重新拼接每个图像并保存
    for image_file, info in block_info.items():
        save_reconstructed_image(image_file, output_folder, block_folder, info, original_sizes[image_file])
        # save_reconstructed_image(image_file, output_folder2, block_folder2, info, original_sizes[image_file])

    print("Image reconstruction completed. Reconstructed images are saved.")

# 调用拼接函数
if __name__ == "__main__":
    #crop()
    reconstruct_main()


