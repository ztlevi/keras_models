import os
from math import ceil
from random import shuffle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow import keras

import net
from definitions import ROOT_DIR
from net import mobilenet_v1


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


def evaluate_age_mobilenet_v1_train_affectnet_model():
    dataset_dir = "/media/ztlevi/HDD/IMDB-WIKI"

    h5f = h5py.File(os.path.join(dataset_dir, "imdb-wiki.h5"), "r")
    X = h5f["img"][:4000]
    Y = h5f["age"][:4000]
    Y = Y / 100

    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_image_normalization()
    # Building Mobilenet
    net, layers = mobilenet_v1.build_mobilenet_v1(1, (224, 224), img_prep)
    net = tflearn.single_unit(net)

    regression = tflearn.regression(
        net, optimizer="adam", loss="mean_square", metric="R2", learning_rate=0.001
    )

    # Training
    model = tflearn.DNN(
        regression,
        tensorboard_verbose=0,
        tensorboard_dir=os.path.join(ROOT_DIR, "outputs/logs"),
        checkpoint_path=os.path.join(
            ROOT_DIR, "outputs/checkpoints/age_mobilenet_v1_imdb_wiki/ckpt"
        ),
    )

    # model.load(os.path.join(ROOT_DIR, "outputs/checkpoints/age_mobilenet_v1_imdb_wiki/ckpt-2500"))
    model.load("/home/ztlevi/Developer/tflearn-models/outputs/checkpoints/age_mobilenet_v1_audience/ckpt-4472")

    # print("Evaluate unbalanced test set...")
    # evaluation = model.evaluate(X, Y)
    # print(evaluation)

    Y_pred_1 = model.predict([X[0]])
    Y_pred_2 = model.predict([X[1000]])

    num_images = X.shape[0]

    batch_size = 64

    # create list of batches to shuffle the data
    batches_list = list(range(int(ceil(float(num_images) / batch_size))))
    shuffle(batches_list)
    # loop over batches
    for x in range(5):
        i = batches_list[x]

        i_s = i * batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * batch_size, num_images])  # index of the last image in this batch

        # read batch images and remove training mean
        images = X[i_s:i_e, ...]

        # read labels and convert to one hot encoding
        seg_labels = Y_pred[i_s:i_e]
        seg_orginal_labels = Y[i_s:i_e]

        plt.figure(figsize=(10, 10))
        for j in range(25):
            plt.subplot(5, 5, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[j], cmap=plt.cm.binary)
            plt.xlabel(seg_labels[j] + seg_orginal_labels[j])

    plt.show()


if __name__ == "__main__":
    evaluate_tut_model()

    # evaluate_age_mobilenet_v1_train_affectnet_model()
