import os

import h5py
import numpy as np
from tensorflow import keras

from dataset import DataGenerator
from dataset.Audience import get_audience_dataset
from dataset.imdb_wiki import get_imdb_wiki_dataset
from definitions import ROOT_DIR


def evaluate_tut_model():
    data = get_audience_dataset()
    addrs = data["addrs"]
    age_labels = data["gender_labels"]

    num_classes = 2
    batch_size = 64
    validation_size = 1000

    # Building Mobilenet

    val_generator = DataGenerator(
        addrs[:validation_size], age_labels[:validation_size], batch_size, num_classes
    )
    # steps_per_epoch = val_generator.n // val_generator.batch_size

    model = keras.models.load_model(os.path.join(ROOT_DIR, "outputs/tut/gender/model.h5"))

    # evaluation = model.evaluate(X, keras.utils.to_categorical(Y), batch_size=128)
    score = model.evaluate_generator(generator=val_generator)
    print(list(zip(model.metrics_names, score)))


def evaluate_imdb_wiki_model():
    data = get_imdb_wiki_dataset()
    addrs = data["addrs"]
    age_labels = data["gender_labels"]

    num_classes = 2
    batch_size = 64
    validation_size = 2500
    checkpoint_path = os.path.join(
        ROOT_DIR, "outputs", "checkpoints", "gender_mobilenet_v1_imdb_wiki", "ckpt-08-0.48.hdf5"
    )

    # Building Mobilenet

    val_generator = DataGenerator(
        addrs[:validation_size], age_labels[:validation_size], batch_size, num_classes
    )
    # steps_per_epoch = val_generator.n // val_generator.batch_size

    model = keras.models.load_model(checkpoint_path)

    # evaluation = model.evaluate(X, keras.utils.to_categorical(Y), batch_size=128)
    score = model.evaluate_generator(generator=val_generator)
    print(list(zip(model.metrics_names, score)))


def evaluate_fine_tuned_audience_model():
    data = get_audience_dataset()
    addrs = data["addrs"]
    age_labels = data["gender_labels"]

    num_classes = 2
    batch_size = 64
    validation_size = 1000
    checkpoint_path = os.path.join(
        ROOT_DIR, "outputs", "checkpoints", "gender_mobilenet_v1_audience", "ckpt-04-0.22.hdf5"
    )

    # Building Mobilenet

    val_generator = DataGenerator(
        addrs[:validation_size], age_labels[:validation_size], batch_size, num_classes
    )
    # steps_per_epoch = val_generator.n // val_generator.batch_size

    model = keras.models.load_model(checkpoint_path)

    # evaluation = model.evaluate(X, keras.utils.to_categorical(Y), batch_size=128)
    score = model.predict_generator(generator=val_generator)
    print(np.argmax(score, axis=1))
    # print(list(zip(model.metrics_names, score)))


if __name__ == "__main__":
    # print("TUT")
    # evaluate_tut_model()

    # print("IMDB_WIKI")
    # evaluate_imdb_wiki_model()

    print("Audience")
    evaluate_fine_tuned_audience_model()
