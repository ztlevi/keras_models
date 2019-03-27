import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import DataGenerator
from dataset.Audience import get_audience_dataset
from dataset.UTKFace import get_utkface_dataset
from definitions import ROOT_DIR
from training.age import mae_pred
from utils import get_latest_checkpoint
from utils.preresiqusites import run_preresiqusites

# Set gpu usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
keras.backend.set_session(tf.Session(config=config))

run_preresiqusites()

################################################################################
# Custom variables
################################################################################
num_classes = 101
batch_size = 64
validation_size = 1000
input_shape = (224, 224, 3)
app_id = "age_mobilenet_v1_utkface"

################################################################################
# Create dataset generator
################################################################################
data = get_utkface_dataset()
addrs = data["addrs"]
age_labels = data["age_labels"]

train_generator = DataGenerator(
    addrs[validation_size:], age_labels[validation_size:], batch_size, num_classes
)
val_generator = DataGenerator(
    addrs[:validation_size], age_labels[:validation_size], batch_size, num_classes
)
steps_per_epoch = train_generator.n // train_generator.batch_size

################################################################################
# Create and load mobilenet
################################################################################
model = keras.applications.mobilenet.MobileNet(
    input_shape=input_shape, weights=None, include_top=False
)
x = model.output
x = keras.layers.GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)
preds = keras.layers.Dense(num_classes, activation="softmax")(x)
model = keras.models.Model(inputs=model.input, outputs=preds)

print("=============================== MODEL DESC =============================")
for i, layer in enumerate(model.layers):
    print(i, layer.name)
print("========================================================================")

################################################################################
# Load checkpoint
################################################################################
checkpoint_path = os.path.join(ROOT_DIR, "outputs", "checkpoints", app_id, "ckpt.h5")
if not os.path.exists(os.path.dirname(checkpoint_path)):
    os.makedirs(os.path.dirname(checkpoint_path))

# Load previous checkpoints
latest_checkpoint = get_latest_checkpoint(
    os.path.join(ROOT_DIR, "outputs", "checkpoints", "age_mobilenet_v1_audience")
)
if os.path.exists(latest_checkpoint):
    model.load_weights(latest_checkpoint)

################################################################################
# Checkpoint and tensorboard callbacks
################################################################################
csv_logger = keras.callbacks.CSVLogger(
    os.path.join(ROOT_DIR, "outputs", "logs", app_id, "log.csv"), append=True, separator=","
)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_mae_pred", verbose=1, save_best_only=True, mode="min", period=1
)

log_path = os.path.join(ROOT_DIR, "outputs", "logs", app_id)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_path, batch_size=batch_size)


################################################################################
# Train
################################################################################
# Adam optimizer
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[mae_pred, "accuracy"])

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    verbose=1,
    validation_data=val_generator,
    shuffle=True,
    use_multiprocessing=True,
    workers=6,
    callbacks=[checkpoint_callback, tensorboard_callback, csv_logger],
)
