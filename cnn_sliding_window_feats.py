from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
import cv2
import math
import numpy as np
from collections import namedtuple


def open_and_prepare_image(image_path, window_size=224, stride=32, steps=(15, 15)):
    image = cv2.imread(image_path)

    target_height = window_size + (steps[0]-1) * stride
    target_width = window_size + (steps[1]-1) * stride

    target_wh_ratio = target_width/target_height

    image_height = image.shape[0]
    image_width = image.shape[1]

    image_wh_ratio = image_width/image_height

    # if image is taller and skinner than target, scale width first then crop height
    # else if image is shorter and fatter than target, scale height first then crop width

    if image_wh_ratio < target_wh_ratio:
        scale_percent = target_width/image.shape[1]
        scale_height = math.floor(scale_percent * image.shape[0])
        image = cv2.resize(image, (target_width, scale_height),
                           interpolation=cv2.INTER_CUBIC)
        m1 = (scale_height - target_height)//2
        m2 = target_height + m1
        return image[m1:m2, :, :]

    else:
        scale_percent = target_height/image.shape[0]
        scale_width = math.floor(scale_percent * image.shape[1])
        image = cv2.resize(image, (scale_width, target_height),
                           interpolation=cv2.INTER_CUBIC)
        m1 = (scale_width-target_width)//2
        m2 = target_width + m1
        return image[:, m1:m2, :]


def extract_windows(image_path, window_size=224, stride=32, steps=(15, 15)):

    image = open_and_prepare_image(image_path, window_size, stride, steps)

    thumbnail = cv2.resize(image, (steps[1] + window_size//32 - 1,
                                   steps[0] + window_size//32 - 1), interpolation=cv2.INTER_CUBIC)

    print("image.shape", image.shape, "thumbnail.shape", thumbnail.shape)

    windows = np.zeros(
        (steps[0] * steps[1], window_size, window_size, image.shape[2]))
    coords = np.zeros((steps[0] * steps[1], 2))

    for i in range(steps[0]):
        for j in range(steps[1]):
            # print((stride*i),(stride*i+window_size), (stride*j),(stride*j+window_size))
            img = image[(stride*i):(stride*i+window_size),
                        (stride*j):(stride*j+window_size)]
            windows[i * steps[1] + j] = img
            coords[i * steps[1] + j] = (i, j)

    # resize image to size of grid?

    return windows, coords, thumbnail


def mean_local_feats(local_feats):

    mean_feat = np.zeros((local_feats[0].shape[0],))

    for i in range(len(local_feats)):
        patch_feat = local_feats[i]
        mean_feat = mean_feat + patch_feat

    mean_feat = mean_feat / len(local_feats)

    return mean_feat


BlockDescriptor = namedtuple(
    "BlockDescriptor", "image coordinates cnn_features color")


def calc_image_descriptors(image_path, model, window_size_blocks=7, block_size=32, block_grid=(21, 28)):
    window_size = window_size_blocks * block_size
    steps = (block_grid[0]-window_size_blocks+1,
             block_grid[1]-window_size_blocks+1)
    stride = block_size

    windows, coords, thumbnail = extract_windows(
        image_path, window_size, stride, steps)

    print("windows.shape", windows.shape)
    print("thumbnail.shape", thumbnail.shape)
    x = preprocess_input(windows)
    y = model.predict(x)
    print(y.shape)

    foo = {}

    for i in range(block_grid[0]):
        for j in range(block_grid[1]):
            foo[(i, j)] = []

    for i in range(y.shape[0]):
        coord = coords[i]
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                foo[(int(coord[0] + j), int(coord[1] + k))].append(y[i, j, k, :])

    descriptors = []

    window_blocks = window_size_blocks**2

    for i in range(block_grid[0]):
        for j in range(block_grid[1]):
            # if len(foo[(i,j)]) == window_blocks:
            descriptor = BlockDescriptor(
                image_path, (i, j), mean_local_feats(foo[(i, j)]), thumbnail[i, j])
            descriptors.append(descriptor)

    return descriptors


def extract_features(image_paths, block_grid=(21, 28)):
    # returns a list of lists of BlockDescriptors

    model = vgg16.VGG16(weights="imagenet", include_top=False,
                        input_shape=(224, 224, 3))

    image_count = len(image_paths)

    result = []

    for i in range(image_count):

        image_path = image_paths[i]
        print(i, image_path)

        descriptors = calc_image_descriptors(
            image_path, model, block_grid=block_grid)

        result.extend(descriptors)

    return result
