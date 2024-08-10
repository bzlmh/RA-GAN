import cv2
import os

# def simple_threshold(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply fixed thresholding
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
#     return binary

# Input and output folder paths
input_folder = '../datasets/HDIBCO/img'
output_folder = '../datasets/img'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith(
            '.tif') or filename.endswith('.tiff') or filename.endswith('.bmp'):
        # Read the image
        image = cv2.imread(os.path.join(input_folder, filename))
        #
        # # Call the simple thresholding algorithm
        # binary_image = simple_threshold(image)

        # Build output file path
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

        # Save the image
        cv2.imwrite(output_path, image)

        print(f'{filename} processing complete')

print('All images processed successfully')
