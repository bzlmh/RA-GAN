import cv2
import os
from tqdm import tqdm
import numpy as np


def create_output_folders(output_folder1, output_folder2):
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)


def process_images(input_folder, prewitt_output_folder, sobel_output_folder):
    create_output_folders(prewitt_output_folder, sobel_output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc='Processing images', unit='image'):
        input_file = os.path.join(input_folder, filename)
        output_file_prewitt = os.path.join(prewitt_output_folder, filename)
        output_file_sobel = os.path.join(sobel_output_folder, filename)

        img = cv2.imread(input_file)
        if img is None:
            print(f"Warning: Failed to read {input_file}. Skipping.")
            continue

        if len(img.shape) == 3:  # If the image has multiple channels, convert it to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:  # If the image is already grayscale
            gray = img
        else:
            print(f"Warning: Unrecognized image format for {input_file}. Skipping.")
            continue

        prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(gray, -1, prewitt_kernel_x)
        prewitt_y = cv2.filter2D(gray, -1, prewitt_kernel_y)

        prewitt_img = cv2.magnitude(prewitt_x.astype(np.float32), prewitt_y.astype(np.float32))

        cv2.imwrite(output_file_prewitt, prewitt_img)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_img = cv2.magnitude(sobel_x, sobel_y)

        cv2.imwrite(output_file_sobel, sobel_img)


if __name__ == "__main__":
    input_folder = "../datasets/Bicklydiary/patch/img"
    prewitt_output_folder = "../datasets/Bicklydiary/patch/prewitt"
    sobel_output_folder = "../datasets/Bicklydiary/patch/sobel"

    process_images(input_folder, prewitt_output_folder, sobel_output_folder)
