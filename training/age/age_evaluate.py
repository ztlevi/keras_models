import os
from random import shuffle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from skimage.io import imread
from tensorflow import keras

from dataset import DataGenerator
from dataset.Audience import get_audience_dataset
from dataset.UTKFace import get_utkface_dataset
from dataset.imdb_wiki import get_imdb_wiki_dataset
from definitions import ROOT_DIR
from training.age import (Linear_1_bias, coral_loss, mae_pred,
                          task_importance_weights)


def evaluate_tut_model():
    dataset_dir = "/media/ztlevi/HDD/IMDB-WIKI"
    h5f = h5py.File(os.path.join(dataset_dir, "imdb-wiki.h5"), "r")
    X = h5f["img"]
    Y = h5f["age"][:4000]

    # dataset_dir = "/media/ztlevi/HDD/AdienceBenchmarkOfUnfilteredFaces"
    # h5f = h5py.File(os.path.join(dataset_dir, "audience.h5"), "r")
    # X = h5f["img"][:3000]
    # Y = h5f["gender"][:3000]

    model = keras.models.load_model(os.path.join(ROOT_DIR, "outputs/tut/age/model.h5"))
    print(model.summary())
    print(model.metrics_names)

    # evaluation = model.evaluate(X, keras.utils.to_categorical(Y), batch_size=128)
    evaluation = model.evaluate(X, Y, batch_size=128)

    print(evaluation)


def evaluate_age_mobilenet_v1_imdb_wiki():
    validation_size = 1000
    data = get_imdb_wiki_dataset()
    addrs = data["addrs"][:validation_size]
    age_labels = data["age_labels"][:validation_size]
    gender_labels = data["gender_labels"][:validation_size]

    num_classes = 101
    batch_size = 64

    imp = task_importance_weights(age_labels, num_classes)
    plt.figure("Weight importance")
    plt.plot(imp)
    plt.show()

    checkpoint_path = os.path.join(
        ROOT_DIR, "outputs", "checkpoints", "age_mobilenet_v1_imdb_wiki", "ckpt.h5"
    )

    # Building Mobilenet

    val_generator = DataGenerator(
        addrs[:validation_size], age_labels[:validation_size], batch_size, num_classes
    )

    # steps_per_epoch = val_generator.n // val_generator.batch_size

    loss = coral_loss(imp)
    model = keras.models.load_model(
        checkpoint_path,
        custom_objects={"loss": loss, "mae_pred": mae_pred, "Linear_1_bias": Linear_1_bias},
    )

    pred = model.predict_generator(generator=val_generator)
    pred = pred > 0.5
    y_pred = np.sum(pred, axis=1)
    mae = np.mean(np.abs(age_labels - y_pred))
    print("mae: {}".format(mae))

    # print(list(zip(model.metrics_names, score)))
    plot(validation_size, batch_size, addrs, gender_labels, age_labels, y_pred)


def evaluate_age_mobilenet_v1_audience():
    validation_size = 1000
    data = get_audience_dataset()
    addrs = data["addrs"][:validation_size]
    age_labels = data["age_labels"][:validation_size]
    gender_labels = data["gender_labels"][:validation_size]

    num_classes = 101
    batch_size = 64
    checkpoint_path = os.path.join(
        ROOT_DIR, "outputs", "checkpoints", "age_mobilenet_v1_audience", "ckpt.h5"
    )

    imp = task_importance_weights(age_labels, num_classes)
    plt.figure("Weight importance")
    plt.plot(imp)
    plt.show()

    # Building Mobilenet

    val_generator = DataGenerator(addrs, age_labels, batch_size, num_classes)
    # steps_per_epoch = val_generator.n // val_generator.batch_size

    loss = coral_loss(imp)
    model = keras.models.load_model(
        checkpoint_path,
        custom_objects={"loss": loss, "mae_pred": mae_pred, "Linear_1_bias": Linear_1_bias},
    )

    # evaluation = model.evaluate(X, keras.utils.to_categorical(Y), batch_size=128)
    pred = model.predict_generator(generator=val_generator)
    pred = pred > 0.5
    y_pred = np.sum(pred, axis=1)
    mae = np.mean(np.abs(age_labels - y_pred))
    print("mae: {}".format(mae))

    plot(validation_size, batch_size, addrs, gender_labels, age_labels, y_pred)


def evaluate_age_mobilenet_v1_utkface():
    validation_size = 1000
    data = get_utkface_dataset()
    addrs = data["addrs"][:validation_size]
    age_labels = data["age_labels"][:validation_size]
    gender_labels = data["gender_labels"][:validation_size]

    num_classes = 101
    batch_size = 64
    checkpoint_path = os.path.join(
        ROOT_DIR, "outputs", "checkpoints", "age_mobilenet_v1_utkface", "ckpt.h5"
    )

    imp = task_importance_weights(age_labels, num_classes)
    plt.figure("Weight importance")
    plt.plot(imp)
    plt.show()

    # Building Mobilenet

    val_generator = DataGenerator(addrs, age_labels, batch_size, num_classes)
    # steps_per_epoch = val_generator.n // val_generator.batch_size

    loss = coral_loss(imp)
    model = keras.models.load_model(
        checkpoint_path,
        custom_objects={"loss": loss, "mae_pred": mae_pred, "Linear_1_bias": Linear_1_bias},
    )

    # evaluation = model.evaluate(X, keras.utils.to_categorical(Y), batch_size=128)
    pred = model.predict_generator(generator=val_generator)
    pred = pred > 0.5
    y_pred = np.sum(pred, axis=1)
    mae = np.mean(np.abs(age_labels - y_pred))
    print("mae: {}".format(mae))

    plot(validation_size, batch_size, addrs, gender_labels, age_labels, y_pred)


def plot(num_images, batch_size, addrs, gender_labels, age_labels, y_pred):
    class_names = ["female", "male"]
    # create list of batches to shuffle the data
    batches_list = list(range(int(ceil(float(num_images) / batch_size))))
    shuffle(batches_list)

    for x in range(5):

        i = batches_list[x]

        i_s = i * batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * batch_size, num_images])  # index of the last image in this batch

        # read batch images and remove training mean
        images = [imread(addr) for addr in addrs[i_s:i_e, ...]]

        # read labels and convert to one hot encoding
        seg_gender = gender_labels[i_s:i_e]
        seg_age = age_labels[i_s:i_e]
        seg_pred = y_pred[i_s:i_e]

        plt.figure(figsize=(10, 10))
        for j in range(25):
            plt.subplot(5, 5, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            if j < len(images):
                plt.imshow(images[j], cmap=plt.cm.binary)
                plt.xlabel(
                    class_names[seg_gender[j]]
                    + "_age_"
                    + str(seg_age[j])
                    + "_pred_"
                    + str(seg_pred[j])
                )
        plt.savefig(os.path.join(ROOT_DIR, "outputs","figures", "evaluation_batch_{}".format(x)))

    plt.show()


if __name__ == "__main__":
    # evaluate_tut_model()
    # evaluate_age_mobilenet_v1_imdb_wiki()
    # evaluate_age_mobilenet_v1_audience()
    evaluate_age_mobilenet_v1_utkface()
