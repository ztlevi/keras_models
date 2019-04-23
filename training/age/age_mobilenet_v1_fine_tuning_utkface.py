import os

import tensorflow as tf
from tensorflow import keras

from dataset.UTKFace import get_utkface_dataset
from definitions import NUM_CPUS, ROOT_DIR, all_args
from training.age import (AgeDataGenerator, Linear_1_bias, coral_loss,
                          mae_pred, task_importance_weights)
from utils.preresiqusites import run_preresiqusites

args = all_args[os.path.splitext(os.path.basename(__file__))[0]]

if args["use_remote"]:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUS
    num_gpus = len(args.GPUS.split(","))
else:
    # Set gpu usage
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    keras.backend.set_session(tf.Session(config=config))

    num_gpus = 1

run_preresiqusites()

################################################################################
# Custom variables
################################################################################
num_classes = 101
batch_size = 64 * num_gpus
validation_size = 1000
input_shape = (224, 224, 3)
app_id = "age_mobilenet_v1_utkface"

################################################################################
# Create dataset generator
################################################################################
data = get_utkface_dataset(args["use_remote"])
addrs = data["addrs"]
age_labels = data["age_labels"]

imp = task_importance_weights(age_labels, num_classes)

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
    input_shape=input_shape, weights=None, include_top=False
)
x = model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.001)(x)
x = keras.layers.Dense(1, use_bias=False)(x)
x = Linear_1_bias(num_classes)(x)

model = keras.models.Model(inputs=model.input, outputs=x)

print("=============================== MODEL DESC =============================")
for i, layer in enumerate(model.layers):
    print(i, layer.name)
print("========================================================================")

if num_gpus > 1:
    model = keras.utils.multi_gpu_model(model, gpus=num_gpus, cpu_merge=True)

################################################################################
# Load checkpoint
################################################################################
# Load IMDB_WIKI checkpoint
imdb_wiki_checkpoint = os.path.join(
    ROOT_DIR, "outputs", "checkpoints", "age_mobilenet_v1_imdb_wiki", "ckpt.h5"
)
if os.path.exists(imdb_wiki_checkpoint):
    model.load_weights(imdb_wiki_checkpoint)

# Load previous checkpoint
checkpoint_path = os.path.join(ROOT_DIR, "outputs", "checkpoints", app_id, "ckpt.h5")
if not os.path.exists(os.path.dirname(checkpoint_path)):
    os.makedirs(os.path.dirname(checkpoint_path))
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)

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

callback_list = [checkpoint_callback, tensorboard_callback, csv_logger]

################################################################################
# Train
################################################################################
# Adam optimizer
opt = keras.optimizers.Adam(lr=0.001)

coral_loss = coral_loss(imp)
model.compile(optimizer=opt, loss=coral_loss, metrics=[mae_pred])

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=40,
    verbose=1,
    validation_data=val_generator,
    shuffle=True,
    use_multiprocessing=True,
    workers=NUM_CPUS,
    callbacks=callback_list,
)
