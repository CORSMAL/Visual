import glob
import itertools
import os
import random
import six
import numpy as np
import cv2

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")


    def tqdm(iter):
        return iter

DATA_LOADER_SEED = 0

random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


class DataLoaderError(Exception):
    pass


def image_resize_no_dist(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if (w > h):
    #     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    #     (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    # if width is None:
    # calculate the ratio of the height and construct the
    # dimensions
    r = height / float(h)
    if (int(w * r) < width):
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    (h, w) = resized.shape[:2]

    delta_w = width - w
    delta_h = height - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    # return the resized image
    return padded


def get_pairs_from_paths(images_path, annotationsHolder):
    """  """
    img_ann_list = []
    for i in range(0, len(glob.glob(os.path.join(images_path,"*.png")))):
        img = os.path.join(images_path, annotationsHolder['annotations'][i]['image_name'])
        ann = annotationsHolder['annotations'][i]['mass']
        img_ann_list.append((img, ann))
    return img_ann_list


def get_image_array(image_input, width, height, imgNorm="divide", ordering='channels_first'):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}".format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        # img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        img = np.float32(image_resize_no_dist(img, width=width, height=height)) / 127.5 - 1
        img = img[:, :, ::-1]
    elif imgNorm == "sub_mean":
        # img = cv2.resize(img, (width, height))
        img = image_resize_no_dist(img, width=width, height=height)
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        # img = cv2.resize(img, (width, height))
        img = image_resize_no_dist(img, width=width, height=height)
        img = img.astype(np.float32)
        img = img / 255.0
        img = img[:, :, ::-1]
    elif imgNorm == "no_op":
        img = image_resize_no_dist(img, width=width, height=height)
        img = img.astype(np.float32)
        img = img[:, :, ::-1]

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def image_labels_generator(images_path, annotationsHolder, batch_size, input_height, input_width):
    img_ann_pairs = get_pairs_from_paths(images_path, annotationsHolder)
    random.shuffle(img_ann_pairs)
    zipped = itertools.cycle(img_ann_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, ann = next(zipped)

            im = cv2.imread(im, 1)
            cv2.imshow("", im)
            cv2.waitKey(10)

            X.append(get_image_array(im, input_width,
                                     input_height, ordering="channels_last"))

            Y.append(ann)

        yield np.array(X), np.array(Y)
