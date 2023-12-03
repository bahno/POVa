import numpy as np

import sys

from tools import collage
from tools import readCIFAR
from matplotlib import pyplot as plt


def create_u_net():
    """
    To allow a seamless tiling of the output segmentation map (see Figure 2), it
    is important to select the input tile size such that all 2x2 max-pooling operations
    are applied to a layer with an even x- and y-size. (https://arxiv.org/pdf/1505.04597.pdf)

    Default example has image 572 x 572, lets test that first
    Model of u-NET neural network

    Note for cropping:
    "The cropping is necessary due to the loss of border pixels in
    every convolution."
    """
    from keras.layers import Input, Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
    from keras.models import Model
    from keras import regularizers

    image_shape = (572, 572, 3)

    ###############
    # Downscaling #
    ###############
    input_size = Input(shape=image_shape)

    conv_l1_1 = Conv2D(64, (3, 3), activation='relu')(input_size)

    conv_l1_2 = Conv2D(64, (3, 3), activation='relu')(conv_l1_1)

    max_pool_l1_l2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_l1_2)

    conv_l2_1 = Conv2D(128, (3, 3), activation='relu')(max_pool_l1_l2)

    conv_l2_2 = Conv2D(128, (3, 3), activation='relu')(conv_l2_1)

    max_pool_l2_l3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_l2_2)

    conv_l3_1 = Conv2D(256, (3, 3), activation='relu')(max_pool_l2_l3)

    conv_l3_2 = Conv2D(256, (3, 3), activation='relu')(conv_l3_1)

    max_pool_l3_l4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_l3_2)

    conv_l4_1 = Conv2D(512, (3, 3), activation='relu')(max_pool_l3_l4)

    conv_l4_2 = Conv2D(512, (3, 3), activation='relu')(conv_l4_1)

    max_pool_l4_l5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_l4_2)

    conv_l5_1 = Conv2D(1024, (3, 3), activation='relu')(max_pool_l4_l5)

    conv_l5_2 = Conv2D(1024, (3, 3), activation='relu')(conv_l5_1)

    #############
    # Upscaling #
    #############
    up_sample_l5_l4 = UpSampling2D(size=(2, 2))(conv_l5_2)

    merge_l4 = concatenate([conv_l4_2, up_sample_l5_l4], axis=3)

    conv_up_l4_1 = Conv2D(1024, (3, 3), activation='relu')(merge_l4)

    conv_up_l4_2 = Conv2D(512, (3, 3), activation='relu')(conv_up_l4_1)

    up_sample_l4_l3 = UpSampling2D(size=(2, 2))(conv_up_l4_2)

    merge_l3 = concatenate([conv_l3_2, up_sample_l4_l3], axis=3)

    conv_up_l3_1 = Conv2D(512, (3, 3), activation='relu')(merge_l3)

    conv_up_l3_2 = Conv2D(256, (3, 3), activation='relu')(conv_up_l3_1)

    up_sample_l3_l2 = UpSampling2D(size=(2, 2))(conv_up_l3_2)

    merge_l2 = concatenate([conv_l2_2, up_sample_l3_l2], axis=3)

    conv_up_l2_1 = Conv2D(256, (3, 3), activation='relu')(merge_l2)

    conv_up_l2_2 = Conv2D(128, (3, 3), activation='relu')(conv_up_l2_1)

    up_sample_l2_l1 = UpSampling2D(size=(2, 2))(conv_up_l2_2)

    merge_l1 = concatenate([conv_l1_2, up_sample_l2_l1], axis=3)

    conv_up_l1_1 = Conv2D(128, (3, 3), activation='relu')(merge_l1)

    conv_up_l1_2 = Conv2D(64, (3, 3), activation='relu')(conv_up_l1_1)

    up_sample_l1_res = UpSampling2D(size=(2, 2))(conv_up_l1_2)

    model = Model(inputs=image_shape, outputs=up_sample_l1_res)

    return model
