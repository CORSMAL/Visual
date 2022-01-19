import keras.layers
import keras.callbacks
import tensorflow as tf
import json
from tensorflow.keras.applications import MobileNetV3Small
from keras.models import Model
import glob
from data_loader import image_labels_generator
from keras_flops import get_flops
import os


def get_callbacks(model_name, log_dir):
    callb = [
        keras.callbacks.ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath=model_name + "_{epoch}.h5",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            period=1,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            profile_batch=0
        ),
        keras.callbacks.History(
        )
    ]
    return callb


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
model.compile(optimizer="adam",
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])
train_images = "C:\\Users\\Tommy\\Downloads\\fold_0\\rgb\\train"
train_size = len(glob.glob(os.path.join(train_images, "*.png")))
train_annotations = "C:\\Users\\Tommy\\Downloads\\annotations_train_0.json"
with open(train_annotations) as json_data_file:
    train_annotationsHolder = json.load(json_data_file)

val_images = "C:\\Users\\Tommy\\Downloads\\fold_0\\rgb\\val"
val_annotations = "C:\\Users\\Tommy\\Downloads\\annotations_val_0.json"
val_size = len(glob.glob(os.path.join(train_images, "*.png")))
with open(val_annotations) as json_data_file:
    val_annotationsHolder = json.load(json_data_file)

batch_size = 10
train_generator = image_labels_generator(images_path=train_images, annotationsHolder=train_annotationsHolder,
                                         batch_size=batch_size,
                                         input_height=input_shape[0], input_width=input_shape[1])
val_generator = image_labels_generator(images_path=val_images, annotationsHolder=val_annotationsHolder,
                                       batch_size=batch_size,
                                       input_height=input_shape[0], input_width=input_shape[1])

# Start training
model.fit(x=train_generator,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          callbacks=[get_callbacks("model", "C:\\Users\\Tommy\\Downloads\\Visual\\Encoder\\log")],
          validation_data=val_generator,
          validation_steps=val_size // batch_size,
          steps_per_epoch=train_size // batch_size,
          use_multiprocessing=False)
