import os
import cv2
import numpy as np
from PIL import Image

def get_otsu(img):
    """
    Apply Otsu's thresholding to an image.

    Parameters:
    img (PIL.Image): Input image

    Returns:
    PIL.Image: Image after Otsu's thresholding
    """
    img_gray = img.convert('L')  # Convert to grayscale
    img_gray_np = np.asarray(img_gray).astype(np.uint8)  # Convert to numpy array
    _, th2 = cv2.threshold(img_gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply Otsu's thresholding
    return Image.fromarray(th2)  # Convert back to PIL image

# Folder paths
img_folder = "../results/A4/c16/"
otsu_folder = "../results/A4/c16/"

# Create otsu_folder if it does not exist
os.makedirs(otsu_folder, exist_ok=True)

# Process each file in img_folder
for filename in os.listdir(img_folder):
    img_path = os.path.join(img_folder, filename)  # Full path to the image
    img = Image.open(img_path)  # Open the image

    # Get Otsu binary image
    otsu_img = get_otsu(img)

    # Save the Otsu binary image
    otsu_save_path = os.path.join(otsu_folder, filename)  # Generate save path
    otsu_img.save(otsu_save_path)  # Save the image

    print(f"Otsu image saved: {otsu_save_path}")
