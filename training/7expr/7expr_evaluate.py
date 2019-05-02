import os

import h5py
import tflearn

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import DataGenerator
from dataset.Affectnet import get_affectnet_dataset
from definitions import ROOT_DIR

class_names = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger",
    "Contempt",
    "None",
    "Uncertain",
    "Non-Face",
]


def evaluate_tut_model():
    num_classes = 7
    batch_size = 64
    validation_size = 2500
    data = get_affectnet_dataset()
    train_addrs = data["training"]["addrs"][:validation_size]
    train_expression_labels = data["training"]["expression_labels"][:validation_size]
    train_expression_labels[train_expression_labels > 6] = 0
    val_addrs = data["validation"]["addrs"]
    val_expression_labels = data["validation"]["expression_labels"]
    val_expression_labels[val_expression_labels > 6] = 0

    val_generator = DataGenerator(train_addrs, train_expression_labels, batch_size, num_classes)

    model = keras.models.load_model("./model.h5")

    evaluation = model.evaluate_generator(val_generator, batch_size=batch_size)
    print(evaluation)


def evaluate_7expr_mobilenet_v1_train_affectnet_model():
    num_classes = 7
    batch_size = 64
    validation_size = 2500
    data = get_affectnet_dataset()
    train_addrs = data["training"]["addrs"][:validation_size]
    train_expression_labels = data["training"]["expression_labels"][:validation_size]
    train_expression_labels[train_expression_labels > 6] = 0
    val_addrs = data["validation"]["addrs"]
    val_expression_labels = data["validation"]["expression_labels"]
    val_expression_labels[val_expression_labels > 6] = 0

    val_generator = DataGenerator(train_addrs, train_expression_labels, batch_size, num_classes)

    checkpoint_path = os.path.join(
        ROOT_DIR, "outputs", "checkpoints", "7expr_mobilenet_v1_affectnet", "ckpt.h5"
    )
    model = keras.models.load_model(checkpoint_path)

    evaluation = model.evaluate_generator(generator=val_generator)
    print(list(zip(model.metrics_names, evaluation)))


if __name__ == "__main__":
    # evaluate_tut_model()

    evaluate_7expr_mobilenet_v1_train_affectnet_model()
