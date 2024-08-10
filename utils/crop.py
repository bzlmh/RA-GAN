import os
import cv2
import numpy as np

def pad_image(image, target_height, target_width):

    height, width = image.shape[:2]

    pad_height = target_height - height
    pad_width = target_width - width

    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    padded_image[:height, :width] = image

    return padded_image

def crop_images(input_folders, output_folders, block_size=(256, 256)):
    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            rows = (height + block_size[1] - 1) // block_size[1]
            cols = (width + block_size[0] - 1) // block_size[0]

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

                    print(f"Saved {output_filename}")

input_folders = ["../datasets/Degardtest/img","../datasets/Degardtest/ostu","../datasets/Degardtest/sobel","../datasets/Degardtest/prewitt"]
output_folders = ["test/img","test/ostu","test/sobel","test/prewitt"]

crop_images(input_folders, output_folders)
