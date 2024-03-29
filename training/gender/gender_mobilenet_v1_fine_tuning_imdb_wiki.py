import os

import tensorflow as tf
from tensorflow import keras

from dataset import DataGenerator
from dataset.imdb_wiki import get_imdb_wiki_dataset
from definitions import NUM_CPUS, ROOT_DIR
from utils.preresiqusites import run_preresiqusites

# Set gpu usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
keras.backend.set_session(tf.Session(config=config))

run_preresiqusites()

################################################################################
# Custom variables
################################################################################
num_classes = 2
batch_size = 64
validation_size = 2500
input_shape = (224, 224, 3)
app_id = "gender_mobilenet_v1_imdb_wiki"

################################################################################
# Create dataset generator
################################################################################
data = get_imdb_wiki_dataset()
addrs = data["addrs"]
age_labels = data["gender_labels"]

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
    input_shape=input_shape, weights="imagenet", include_top=False
)

# Freeze previous layers
for layer in model.layers:
    layer.trainable = False

x = model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.001)(x)
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
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

################################################################################
# Checkpoint and tensorboard callbacks
################################################################################
# checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     checkpoint_path, monitor="val_loss", verbose=1, save_best_only=False, period=1
# )
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max", period=1
)

log_path = os.path.join(ROOT_DIR, "outputs", "logs", app_id)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_path, batch_size=batch_size)

################################################################################
# Train
################################################################################
# Adam optimizer
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=3,
    verbose=1,
    validation_data=val_generator,
    shuffle=True,
    use_multiprocessing=True,
    workers=NUM_CPUS,
    callbacks=[checkpoint_callback, tensorboard_callback],
)

# Unfreeze previous layers
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    verbose=1,
    validation_data=val_generator,
    shuffle=True,
    use_multiprocessing=True,
    workers=NUM_CPUS,
    callbacks=[checkpoint_callback, tensorboard_callback],
    initial_epoch=3,
)
