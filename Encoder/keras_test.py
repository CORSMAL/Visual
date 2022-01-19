import keras.layers
import keras.callbacks
import tensorflow as tf
import json
from tensorflow.keras.applications import MobileNetV3Small
from keras.models import Model
import glob
from data_loader import get_image_array
from keras_flops import get_flops
import os
import cv2
import numpy as np


def get_regression_model(back_model):
    inputs = back_model.input
    avg_pool = back_model.get_layer("global_average_pooling2d").output
    x = keras.layers.Conv2D(filters=256,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='valid', activation="relu")(avg_pool)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(filters=1,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='valid')(x)
    x = keras.layers.Flatten()(x)
    model = Model(inputs=inputs, outputs=x)
    return model


input_shape = [224, 224, 3]
back_model = MobileNetV3Small(input_shape=input_shape,
                              alpha=1.0,
                              minimalistic=True,
                              include_top=True,
                              weights='imagenet',
                              input_tensor=None,
                              classes=1000,
                              pooling='max',
                              dropout_rate=0.2,
                              classifier_activation='softmax')
model = get_regression_model(back_model)
model.summary()
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.05} G")
model.load(".h5")

val_images = "C:\\Users\\Tommy\\Downloads\\fold_0\\rgb\\val"
val_annotations = "C:\\Users\\Tommy\\Downloads\\annotations_val_0.json"
val_size = len(glob.glob(os.path.join(val_images, "*.png")))
with open(val_annotations) as json_data_file:
    val_annotationsHolder = json.load(json_data_file)

min_mass = 2.0
max_mass = 134.0
for i in range(0, len(glob.glob(os.path.join(val_images, "*.png")))):
    true = val_annotationsHolder['annotations'][i]['mass']
    img = cv2.imread(os.path.join(val_images, val_annotationsHolder['annotations'][i]['image_name']))
    img = get_image_array(img, 224, 224, ordering="channels_last")
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    pred = max(0, pred*(max_mass-min_mass) + min_mass)
    print("True: {} \t Pred: {}".format(true, pred))
    print("Diff  = {}".format(true - pred))

