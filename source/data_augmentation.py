import cv2
import os

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def create_stripes(width: int, height: int, strip_width=10) -> Image:
    # create image to draw on
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Draw diagonal stripes
    for i in range(0, width + height, 2 * strip_width):
        draw.line([(i, -5), (-5, i)], fill="#AAFF00", width=strip_width)
        draw.line([(i + strip_width, -5), (-5, i + strip_width)], fill="purple", width=strip_width)

    return image


def blend_image_mask(image, mask):
    # open image and its segmentation mask
    normal_image = Image.open(image).convert("RGB")
    segmentation_mask = Image.open(mask).convert("RGB")

    striped_image = create_stripes(normal_image.width, normal_image.height).convert("RGB")

    # Convert images to numpy arrays for easier manipulation
    normal_array = np.array(normal_image)
    segmentation_array = np.array(segmentation_mask)
    striped_image_array = np.array(striped_image)

    # Create a new array based on your conditions
    new_image_array = np.where((segmentation_array == [0, 0, 0]).all(axis=2, keepdims=True), normal_array,
                               striped_image_array)

    # Convert the new array back to an image
    new_image = Image.fromarray(new_image_array.astype('uint8'))

    return new_image


def merge_all_in_folder(folder_path_in: str):
    import os
    folder_path = folder_path_in + "/Merged"
    # Check if the folder already exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")

    folder_path = folder_path_in + "/JPEGImages"

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            """
                1) Find corresponding segmentation image
                2) Perform merge
                3) Yield result 
            """
            import re
            mask_path = re.sub('jpg$', 'png', file_path.replace("JPEGImages", "Annotations"))
            image_path = file_path

            merge = blend_image_mask(image_path, mask_path)
            merged_path = file_path.replace("JPEGImages", "Merged")
            try:
                merge.save(merged_path)
            except FileNotFoundError:
                dir_path = "/".join(merged_path.split("\\")[:-1])
                os.makedirs(dir_path)


if __name__ == "__main__":
    merge_all_in_folder("../datasets/Davis/train480p/DAVIS")
