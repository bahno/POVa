import numpy as np
from matplotlib import pyplot as plt


def create_u_net():
    """
    To allow a seamless tiling of the output segmentation map (see Figure 2), it
    is important to select the input tile size such that all 2x2 max-pooling operations
    are applied to a layer with an even x- and y-size. (https://arxiv.org/pdf/1505.04597.pdf)

    TODO Default example has image 572 x 572, lets test that first
    Model of u-NET neural network

    TODO cropping:
    "The cropping is necessary due to the loss of border pixels in
    every convolution."

    For example, if you have a 3x3 convolutional layer followed by a 2x2 max pooling layer,
    you would lose (3 - 1) + 2 = 4 pixels on each side. If you repeat this pattern,
    the total reduction will be the sum of these losses.

    """
    from keras.layers import Input, Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate
    from keras.models import Model
    from keras import regularizers

    image_shape = (572, 572, 3)

    ####################
    # Contractive path #
    ####################
    input_size = Input(shape=image_shape)

    # L1
    conv_l1_1 = Conv2D(64, (3, 3), activation='relu')(input_size)

    conv_l1_2 = Conv2D(64, (3, 3), activation='relu')(conv_l1_1)

    max_pool_l1_l2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_l1_2)

    # L2
    conv_l2_1 = Conv2D(128, (3, 3), activation='relu')(max_pool_l1_l2)

    conv_l2_2 = Conv2D(128, (3, 3), activation='relu')(conv_l2_1)

    max_pool_l2_l3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_l2_2)

    # L3
    conv_l3_1 = Conv2D(256, (3, 3), activation='relu')(max_pool_l2_l3)

    conv_l3_2 = Conv2D(256, (3, 3), activation='relu')(conv_l3_1)

    max_pool_l3_l4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_l3_2)

    # L4
    conv_l4_1 = Conv2D(512, (3, 3), activation='relu')(max_pool_l3_l4)

    conv_l4_2 = Conv2D(512, (3, 3), activation='relu')(conv_l4_1)

    max_pool_l4_l5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_l4_2)

    # L5
    conv_l5_1 = Conv2D(1024, (3, 3), activation='relu')(max_pool_l4_l5)

    conv_l5_2 = Conv2D(1024, (3, 3), activation='relu')(conv_l5_1)

    ##################
    # Expansive path #
    ##################
    up_conv_l5_l4 = Conv2DTranspose(512, kernel_size=(2, 2), strides=2)(conv_l5_2)

    # L4
    crop_l4_2 = Cropping2D(cropping=4)(conv_l4_2)
    merge_l4 = concatenate([crop_l4_2, up_conv_l5_l4], axis=3)

    conv_exp_l4_1 = Conv2D(512, (3, 3), activation='relu')(merge_l4)

    conv_exp_l4_2 = Conv2D(512, (3, 3), activation='relu')(conv_exp_l4_1)

    up_conv_l4_l3 = Conv2DTranspose(256, kernel_size=(2, 2), strides=2)(conv_exp_l4_2)

    # L3
    crop_l3 = Cropping2D(cropping=16)(conv_l3_2)
    merge_l3 = concatenate([crop_l3, up_conv_l4_l3], axis=3)

    conv_exp_l3_1 = Conv2D(256, (3, 3), activation='relu')(merge_l3)

    conv_exp_l3_2 = Conv2D(256, (3, 3), activation='relu')(conv_exp_l3_1)

    up_conv_l3_l2 = Conv2DTranspose(128, kernel_size=(2, 2), strides=2)(conv_exp_l3_2)

    # L2
    crop_l2 = Cropping2D(cropping=40)(conv_l2_2)
    merge_l2 = concatenate([crop_l2, up_conv_l3_l2], axis=3)

    conv_exp_l2_1 = Conv2D(128, (3, 3), activation='relu')(merge_l2)

    conv_exp_l2_2 = Conv2D(128, (3, 3), activation='relu')(conv_exp_l2_1)

    up_conv_l2_l1 = Conv2DTranspose(64, kernel_size=(2, 2), strides=2)(conv_exp_l2_2)

    # L1
    crop_l1 = Cropping2D(cropping=88)(conv_l1_2)
    merge_l1 = concatenate([crop_l1, up_conv_l2_l1], axis=3)

    conv_exp_l1_1 = Conv2D(64, (3, 3), activation='relu')(merge_l1)

    conv_exp_l1_2 = Conv2D(64, (3, 3), activation='relu')(conv_exp_l1_1)

    conv_exp_l1_1x1 = Conv2D(2, (1, 1))(conv_exp_l1_2)

    model = Model(inputs=input_size, outputs=conv_exp_l1_1x1)

    return model


def prepareData(downsample=1):
    # This reads the dataset.
    trnData, tstData, trnLabels, tstLabels = readCIFAR(
        './data/cifar-10-batches-py')
    print('\nDataset tensors')
    print('Training shapes: ', trnData.shape, trnLabels.shape)
    print('Testing shapes: ', tstData.shape, tstLabels.shape)
    print()

    # Convert images from RGB to BGR
    trnData = trnData[::downsample, :, :, ::-1]
    tstData = tstData[::downsample, :, :, ::-1]
    trnLabels = trnLabels[::downsample]
    tstLabels = tstLabels[::downsample]

    # Normalize data
    # This maps all values in trn. and tst. data to range <-0.5,0.5>.
    # Some kind of value normalization is preferable to provide
    # consistent behavior accross different problems and datasets.
    trnData = trnData.astype(np.float32) / 255.0 - 0.5
    tstData = tstData.astype(np.float32) / 255.0 - 0.5
    return trnData, tstData, trnLabels, tstLabels


def main():
    model = create_u_net()
    print('Model summary:')
    model.summary()
    return

    from keras import optimizers
    from keras import losses
    from keras import metrics
    # Use SparseCategoricalCrossentropy loss and Adam optimizer with learning rate 0.001.
    # All the imports you need are provided above.
    model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=[metrics.sparse_categorical_accuracy])

    trnData, tstData, trnLabels, tstLabels = prepareData()

    # Show first 144 images from each set.
    trnCollage = collage(trnData[:144] + 0.5)
    tstCollage = collage(tstData[:144] + 0.5)
    plt.imshow(trnCollage)
    plt.title('Training data')
    plt.show()
    plt.imshow(tstCollage)
    plt.title('Testing data')
    plt.show()

    # Train the network for 5 epochs on mini-batches of 64 images.
    model.fit(
        x=trnData, y=trnLabels,
        batch_size=64, epochs=5, verbose=1,
        validation_data=(tstData, tstLabels), shuffle=True)

    # Save the network:
    model.save('models.h5')

    # Compute network predictions for the test set and show results.
    print('Compute models predictions for test images and display the results.')

    dataToTest = tstData[::20]

    # Compute network (models) responses for dataToTest inputs.
    # This should produce a 2D tensor of the 10 class probabilites for each
    # image in dataToTest. The subsequent code displays the predicted classes.
    classProb = model.predict(dataToTest)

    print('Prediction shape:', classProb.shape)

    # These are the classes as defined in CIFAR-10 dataset in the correct order
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

    # Get the most probable class for each test image.
    predicted_classes = np.argmax(classProb, axis=1)
    for i in range(classProb.shape[1]):
        # Get all images assigned to class "i" and show them.
        class_images = dataToTest[predicted_classes == i]
        if class_images.shape[0]:
            class_collage = collage(class_images)
            title = 'Predicted class {} - {}'.format(i, classes[i])
            plt.imshow(class_collage + 0.5)
            plt.title(title)
            plt.show()

    print('Evaluate network error outside of training on test data.')
    loss, acc = model.evaluate(x=tstData, y=tstLabels, batch_size=64)
    print()
    print('Test loss', loss)
    print('Test accuracy', acc)


if __name__ == "__main__":
    main()
