import posixpath
import random
import struct
from typing import Tuple

import cv2
import os
import re
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import Compose


class DataAugmenter:
    def __init__(self):
        self.transforms = None
        self.stripes = None
        self.prev_merged = None
        self.prev_img = None
        self.prev_mask = None
        self.curr_img = None

        self.flip = random.random() > 0.5
        self.rotate = random.randint(0, 10)
        self.perspective_distort = random.random() > 0.5
        self.perspective_prob = random.random() > 0.5
        self.crop_flag = random.random() > 0.5

    def recalculate_params(self):
        self.flip = random.random() > 0.5
        self.rotate = random.randint(0, 10)
        self.perspective_distort = random.random() > 0.5
        self.perspective_prob = random.random() > 0.5
        self.crop_flag = random.random() > 0.5

    def set_prev_images(self, prev_image_path: str, prev_mask_path: str):
        self.prev_img = Image.open(prev_image_path)
        self.prev_mask = Image.open(prev_mask_path)

    def set_curr_img(self, curr_image_path: str):
        self.curr_img = Image.open(curr_image_path)

    def merge_image_and_mask(self):
        if self.prev_img is None or self.prev_mask is None:
            raise Exception("Previous image and mask must be set before calling this function")

        # open image and its segmentation mask
        striped_image = self.create_striped_image(self.prev_img.width, self.prev_img.height).convert("RGB")

        # Convert images to numpy arrays for easier manipulation
        normal_array = np.array(self.prev_img)
        segmentation_array = np.array(self.prev_mask)
        striped_image_array = np.array(striped_image)

        # Create a new array based on your conditions
        new_image_array = np.where((segmentation_array == [0, 0, 0]).all(axis=2, keepdims=True), normal_array,
                                   striped_image_array)

        # Convert the new array back to an image
        self.prev_merged = Image.fromarray(new_image_array.astype('uint8'))

    def create_striped_image(self, width: int, height: int, stripe_width=10):
        # create image to draw on
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Draw diagonal stripes
        for i in range(0, width + height, 2 * stripe_width):
            draw.line([(i, -5), (-5, i)], fill="#AAFF00", width=stripe_width)
            draw.line([(i + stripe_width, -5), (-5, i + stripe_width)], fill="purple", width=stripe_width)

        self.stripes = image

    def get_N_random_transforms(self, n: int) -> tuple[Compose, Compose]:
        # Create a list of transformations
        geometric_transform_list = list()
        other_transform_list = list()

        # Geometric transformations
        geometric_transform_list.append(transforms.RandomHorizontalFlip(self.flip))
        geometric_transform_list.append(transforms.RandomRotation(self.rotate))
        #geometric_transform_list.append(
        #    transforms.RandomPerspective(distortion_scale=self.perspective_distort,
        #                                 p=self.perspective_prob))
        if self.crop_flag:
            geometric_transform_list.append(transforms.CenterCrop(256))

        # Color space transformations
        other_transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5))

        # Noise injections - Gaussian blur
        other_transform_list.append(transforms.GaussianBlur(kernel_size=(5, 5),
                                                            sigma=(
                                                                0.01,
                                                                0.05)))

        # Greyscale transformation
        other_transform_list.append(transforms.RandomGrayscale(p=0.2))

        # Create a custom transformation that applies a random subset of the above transformations
        # Choose N transformations at random
        chosen_geometric_transforms = random.sample(geometric_transform_list, k=2)
        other_transforms = random.sample(other_transform_list, k=n - 2)

        self.transforms = (
            transforms.Compose(chosen_geometric_transforms),
            transforms.Compose(other_transforms)
        )


def merge_all_in_folder(folder_path_in: str):
    folder_path = folder_path_in + "/Merged"

    # Check if the folder already exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = folder_path_in + "/JPEGImages"

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            mask_path = re.sub('jpg$', 'png', file_path.replace("JPEGImages", "Annotations"))
            image_path = file_path

            data_augmenter = DataAugmenter()
            data_augmenter.set_prev_images(image_path, mask_path)

            data_augmenter.merge_image_and_mask()
            merged_path = file_path.replace("JPEGImages", "Merged")
            try:
                data_augmenter.prev_merged.save(merged_path)
            except FileNotFoundError:
                if sys.platform.startswith('linux'):
                    dir_path = "/".join(merged_path.split("/")[:-1])
                elif sys.platform.startswith('win32'):
                    dir_path = "/".join(merged_path.split("\\")[:-1])
                os.makedirs(dir_path)


def process_directory(directory, data_augmenter=None):

    for filename in os.listdir(directory):

        full_path = posixpath.join(directory, filename)

        if os.path.isfile(full_path):

            # check if folder for newly created files exists
            new_file_path = full_path.replace("JPEGImages", "Augmented")
            if not os.path.exists(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))

            # augment image
            image = Image.open(full_path)
            annotation_image = Image.open(full_path.replace("JPEGImages", "Annotations").replace("jpg", "png"))

            transformed_image = data_augmenter.transforms[1](image)
            transformed_image = data_augmenter.transforms[0](transformed_image)
            transformed_annotation_image = data_augmenter.transforms[0](annotation_image)

            plt.title('Transformed Image')
            plt.imshow(transformed_image)
            plt.axis('off')
            plt.show()

            plt.title('Transformed Annotation Image')
            plt.imshow(transformed_annotation_image)
            plt.axis('off')
            plt.show()

            exit(0)

            # save image to the new folder with the same name
        elif os.path.isdir(full_path):
            # Process directory
            data_augmenter.recalculate_params()
            data_augmenter.get_N_random_transforms(4)
            process_directory(full_path, data_augmenter)


if __name__ == "__main__":
    #process_directory("../datasets/Davis/train480p/DAVIS/JPEGImages/480p", DataAugmenter())
    merge_all_in_folder("../datasets/Davis/train480p/DAVIS")
