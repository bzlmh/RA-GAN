import os
import cv2
import numpy as np


def add_alpha_channel(image):

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 3:
        alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 255
        image = np.concatenate((image, alpha_channel), axis=2)
    return image


def reconstruct_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    block_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

    if not block_files:
        print("No image blocks found in the input folder.")
        return

    row_col_map = {}
    for block_file in block_files:
        parts = os.path.splitext(block_file)[0].split('_')
        filename = parts[0]
        row_col = tuple(map(int, parts[1:3]))
        if filename not in row_col_map:
            row_col_map[filename] = []
        row_col_map[filename].append(row_col)

    for filename, row_col_list in row_col_map.items():
        max_row = max(row for row, _ in row_col_list) + 1
        max_col = max(col for _, col in row_col_list) + 1
        reconstructed_image = None

        for row, col in row_col_list:
            block_path = os.path.join(input_folder, f"{filename}_{row}_{col}.png")
            block = cv2.imread(block_path, cv2.IMREAD_UNCHANGED)

            block = add_alpha_channel(block)

            if reconstructed_image is None:
                block_height, block_width = block.shape[:2]
                reconstructed_image = np.zeros((block_height * max_row, block_width * max_col, 4), dtype=np.uint8)

            x_start = col * block.shape[1]
            x_end = x_start + block.shape[1]
            y_start = row * block.shape[0]
            y_end = y_start + block.shape[0]

            reconstructed_image[y_start:y_end, x_start:x_end] = block

        output_image_path = os.path.join(output_folder, f"{filename}.png")
        cv2.imwrite(output_image_path, reconstructed_image)

        print(f"Reconstructed image saved: {output_image_path}")


input_folder = "mask_out"
output_folder = "1"

reconstruct_images(input_folder, output_folder)
