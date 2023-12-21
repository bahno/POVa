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
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomPerspective, CenterCrop, ColorJitter, \
    Compose, GaussianBlur, RandomGrayscale


class ControlledDistortion:
    def __init__(self):
        self.all_transforms = list()
        self.geometrical_transforms = list()

        self.flip = random.random() > 0.5
        self.rotate = random.randint(0, 10)
        self.perspective_distort = (random.random() > 0.5) * random.uniform(0, 0.5)
        self.crop_flag = random.random() > 0.5
        self.greyscale = random.random() > 0.5
        self.brighness = random.uniform(0.5, 1)
        self.contrast = random.uniform(0.5, 1)

    def transform_image_and_mask(self, img, mask):
        return (
            (Compose(self.all_transforms)(img)),
            Compose(self.geometrical_transforms)(mask)
        )

    def recalculate_params(self):
        self.flip = random.random() > 0.5
        self.rotate = random.randint(0, 10)
        self.perspective_distort = (random.random() > 0.5) * random.uniform(0, 0.5)
        self.crop_flag = random.random() > 0.5

    def randomize_transforms(self):
        print("Randomizing transforms")
        # create new random parameters for the transforms
        self.recalculate_params()

        # Create a list of transformations
        geometric_transform_list = list()
        other_transform_list = list()

        # Geometric transformations
        geometric_transform_list.append(RandomHorizontalFlip(self.flip))
        geometric_transform_list.append(RandomRotation((self.rotate, self.rotate)))
        # geometric_transform_list.append(perspective(distortion_scale=self.perspective_distort, p=1))

        if self.crop_flag:
            geometric_transform_list.append(CenterCrop(256))

        # Color space transformations
        other_transform_list.append(
            ColorJitter(brightness=(self.brighness, self.brighness), contrast=(self.contrast, self.contrast)))
        other_transform_list.append(GaussianBlur(kernel_size=(5, 5), sigma=0.05))
        other_transform_list.append(RandomGrayscale(self.greyscale))

        # Create a custom transformation that applies a random subset of the above transformations
        self.geometrical_transforms = geometric_transform_list
        self.all_transforms = geometric_transform_list + other_transform_list


class DataAugmenter:
    def __init__(self):
        self.distorter = ControlledDistortion()

        self.stripes = None
        self.prev_merged = None
        self.prev_img = None
        self.prev_mask = None
        self.curr_img = None

    def set_prev_images(self, prev_image, prev_mask):
        self.prev_img = prev_image
        self.prev_mask = prev_mask

    def set_curr_img(self, curr_image_path: str):
        self.curr_img = Image.open(curr_image_path)

    def merge_image_and_mask(self):
        if self.prev_img is None or self.prev_mask is None:
            raise Exception("Previous image and mask must be set before calling this function")

        # open image and its segmentation mask
        self.create_striped_image(self.prev_img.width, self.prev_img.height)

        # Convert images to numpy arrays for easier manipulation
        normal_array = np.array(self.prev_img)
        segmentation_array = np.array(self.prev_mask)[:, :, np.newaxis]
        striped_image_array = np.array(self.stripes)

        # Create a new array based on your conditions
        new_image_array = np.where((segmentation_array == 0),
                                   normal_array,
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
            new_image_path = full_path.replace("JPEGImages", "AugmentedJPEGImages")
            new_merge_path = full_path.replace("JPEGImages", "AugmentedMerged")

            if not os.path.exists(os.path.dirname(new_image_path)):
                os.makedirs(os.path.dirname(new_image_path))

            if not os.path.exists(os.path.dirname(new_merge_path)):
                os.makedirs(os.path.dirname(new_merge_path))

            # augment image
            image = transforms.ToTensor()(Image.open(full_path))
            annotation_image = transforms.ToTensor()(
                Image.open(full_path.replace("JPEGImages", "Annotations").replace("jpg", "png")))

            transformed_image, transformed_annotation_image = data_augmenter.distorter.transform_image_and_mask(image,
                                                                                                                annotation_image)

            # save image to the new folder with the same name
            data_augmenter.set_prev_images(transforms.ToPILImage()(transformed_image),
                                           transforms.ToPILImage()(transformed_annotation_image))
            data_augmenter.merge_image_and_mask()

            data_augmenter.prev_merged.save(new_merge_path)
            data_augmenter.prev_img.save(new_image_path)

        elif os.path.isdir(full_path):
            # Process directory
            data_augmenter.distorter.randomize_transforms()
            process_directory(full_path, data_augmenter)


if __name__ == "__main__":
    process_directory("../datasets/Davis/train480p/DAVIS/JPEGImages/480p", DataAugmenter())
    # merge_all_in_folder("../datasets/Davis/train480p/DAVIS")
