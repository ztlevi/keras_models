import os

from tensorflow import keras

from dataset import DataGenerator
from dataset.imdb_wiki import get_imdb_wiki_dataset
from definitions import ROOT_DIR
from utils.preresiqusites import run_preresiqusites

num_classes = 2
batch_size = 64
app_id = "gender_mobilenet_v1_imdb_wiki"

run_preresiqusites()

model = keras.applications.mobilenet.MobileNet(weights="imagenet", include_top=False)
x = model.output
x = keras.layers.GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)
preds = keras.layers.Dense(num_classes, activation="softmax")(x)
model = keras.models.Model(inputs=model.input, outputs=preds)

print("=============================== MODEL DESC =============================")
for i, layer in enumerate(model.layers):
    print(i, layer.name)
print("========================================================================")

data = get_imdb_wiki_dataset()
addrs = data["addrs"]
age_labels = data["gender_labels"]

validation_size = 2500
train_generator = DataGenerator(addrs[validation_size:], age_labels[validation_size:], batch_size, num_classes)
val_generator = DataGenerator(addrs[:validation_size], age_labels[:validation_size], batch_size, num_classes)

checkpoint_path = os.path.join(
    ROOT_DIR, "outputs", "checkpoints", app_id, "ckpt-{epoch:02d}-{val_loss:.2f}.hdf5"
)
if not os.path.exists(os.path.dirname(checkpoint_path)):
    os.makedirs(os.path.dirname(checkpoint_path))

# Load previous checkpoints
previous_checkpoint_path = os.path.join(
    ROOT_DIR, "outputs", "checkpoints", app_id, "ckpt-05-4.40.hdf5"
)
if os.path.exists(previous_checkpoint_path):
    model.load_weights(previous_checkpoint_path)

opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# Adam optimizer
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_loss", verbose=1, save_best_only=False, period=1
)
# checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     checkpoint_path, monitor="val_acc", verbose=1, save_best_only=True, mode='max', period=1
# )

log_path = os.path.join(ROOT_DIR, "outputs", "logs", app_id)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_path, batch_size=batch_size)

steps_per_epoch = train_generator.n // train_generator.batch_size

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    verbose=1,
    validation_data=val_generator,
    validation_steps=steps_per_epoch,
    shuffle=True,
    use_multiprocessing=True,
    workers=6,
    callbacks=[checkpoint_callback, tensorboard_callback],
)
