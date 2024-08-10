import os
import cv2
import numpy as np

def pad_image(image, target_height, target_width):
    """
    Pad the image to the target size.
    """
    height, width = image.shape[:2]

    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    padded_image[:height, :width] = image

    return padded_image

def crop_images(input_folders, output_folders, block_size=(256, 256)):
    """
    Crop images into blocks and save block information and original image size.
    """
    block_info = {}
    original_sizes = {}

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

    np.save("HDIBCO/block_info.npy", block_info)
    np.save("HDIBCO/original_sizes.npy", original_sizes)
    print("Block and original size information saved")

    return block_info, original_sizes

def reconstruct_image(image_file, block_info, original_size, block_folder, block_size=(256, 256)):
    """
    Reconstruct the image from blocks and crop to the original size.
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

    original_height, original_width = original_size
    reconstructed_image = reconstructed_image[:original_height, :original_width]

    return reconstructed_image

def save_reconstructed_image(image_file, output_folder, block_folder, block_info, original_size, block_size=(256, 256)):
    """
    Save the reconstructed image.
    """
    reconstructed_image = reconstruct_image(image_file, block_info, original_size, block_folder, block_size)
    output_filename = f"{image_file}"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, reconstructed_image)
    print(f"Saved reconstructed image {output_filename}")

def reconstruct_main():
    block_folder = "../results/c16"
    output_folder = "../results/A4/c16"

    os.makedirs(output_folder, exist_ok=True)

    block_info = np.load("HDIBCO/block_info.npy", allow_pickle=True).item()
    original_sizes = np.load("HDIBCO/original_sizes.npy", allow_pickle=True).item()

    for image_file, info in block_info.items():
        save_reconstructed_image(image_file, output_folder, block_folder, info, original_sizes[image_file])

    print("Image reconstruction completed. Reconstructed images are saved.")

if __name__ == "__main__":
    reconstruct_main()
