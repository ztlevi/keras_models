import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from dataset.imdb_wiki import get_imdb_wiki_dataset
from definitions import ROOT_DIR
from training import step_decay
from training.age import (AgeDataGenerator, Linear_1_bias, mae_pred,
                          task_importance_weights)
from utils import get_latest_checkpoint
from utils.preresiqusites import run_preresiqusites

# tf.enable_eager_execution()
num_gpus = 4
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

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
validation_size = 2500
input_shape = (224, 224, 3)
app_id = "age_mobilenet_v1_imdb_wiki"

################################################################################
# Create dataset generator
################################################################################
data = get_imdb_wiki_dataset()
addrs = data["addrs"]
age_labels = data["age_labels"]

imp = task_importance_weights(age_labels)
imp = imp[0 : num_classes - 1]

train_generator = AgeDataGenerator(
    addrs[validation_size:], age_labels[validation_size:], batch_size, num_classes
)
val_generator = AgeDataGenerator(
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
x = keras.layers.Dense(1, use_bias=False)(x)
preds = Linear_1_bias(num_classes)(x)

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
latest_checkpoint = get_latest_checkpoint(os.path.dirname(checkpoint_path))
if os.path.exists(latest_checkpoint):
    model.load_weights(latest_checkpoint)

################################################################################
# Checkpoint and tensorboard callbacks
################################################################################
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_mae_pred", verbose=1, save_best_only=True, mode="min", period=1
)

log_path = os.path.join(ROOT_DIR, "outputs", "logs", app_id)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_path, batch_size=batch_size)

# lrate = keras.callbacks.LearningRateScheduler(step_decay)

callback_list = [checkpoint_callback, tensorboard_callback]

################################################################################
# Train
################################################################################
# Adam optimizer
epochs = 50
opt = keras.optimizers.Adam(lr=0.001)


def coral_loss(imp):
    def loss(levels, logits):
        val = -K.sum(
            (K.log(K.sigmoid(logits)) * levels + (K.log(K.sigmoid(logits)) - logits) * (1 - levels))
            * tf.convert_to_tensor(imp, dtype=tf.float32),
            axis=1,
        )
        return K.mean(val)

    return loss


coral_loss = coral_loss(imp)
model.compile(optimizer=opt, loss=coral_loss, metrics=[mae_pred])

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=5,
    verbose=1,
    validation_data=val_generator,
    shuffle=True,
    use_multiprocessing=True,
    workers=6,
    callbacks=callback_list,
)

# Unfreeze previous layers
for layer in model.layers:
    layer.trainable = True

# Load last best checkpoint
latest_checkpoint = get_latest_checkpoint(os.path.dirname(checkpoint_path))
if os.path.exists(latest_checkpoint):
    model.load_weights(latest_checkpoint)

model.compile(optimizer=opt, loss=coral_loss, metrics=[mae_pred])

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    validation_data=val_generator,
    shuffle=True,
    use_multiprocessing=True,
    workers=6,
    callbacks=callback_list,
    initial_epoch=5,
)

last_checkpoint_path = os.path.join(ROOT_DIR, "outputs", "checkpoints", app_id, "last-ckpt.h5")
model.save(last_checkpoint_path)
