# https://stackoverflow.com/questions/51858203/cant-import-frozen-graph-with-batchnorm-layer?noredirect=1&lq=1
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io

from definitions import ROOT_DIR, all_args
from training.age import Linear_1_bias, coral_loss, mae_pred


def freeze_graph(graph, session, output, output_path):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output
        )
        graph_io.write_graph(graphdef_frozen, output_path, "frozen_model.pb", as_text=False)


def load_general_model(appid):
    model = keras.models.load_model(
        os.path.join(ROOT_DIR, "outputs", "checkpoints", appid, "ckpt.h5")
    )
    return model


def load_age_model():
    loss = coral_loss(np.ones(101 - 1))
    checkpoint_path = os.path.join(ROOT_DIR, "outputs/checkpoints/age_mobilenet_v1_utkface/ckpt.h5")
    model = keras.models.load_model(
        checkpoint_path,
        custom_objects={"loss": loss, "mae_pred": mae_pred, "Linear_1_bias": Linear_1_bias},
    )
    return model


def load_tut_age_model():
    model = keras.models.load_model(
        os.path.join(ROOT_DIR, "outputs/tut/age/model.h5"),
        custom_objects={
            "relu6": keras.layers.ReLU(6.0),
            "GlorotUniform": keras.initializers.glorot_uniform(),
            "DepthwiseConv2D": keras.layers.DepthwiseConv2D,
        },
    )
    return model


if __name__ == "__main__":
    keras.backend.set_learning_phase(0)  # this line most important
    keras_load_model = {
        "general": load_general_model("7expr_mobilenet_v1_affectnet"),
        "age": load_age_model,
        "tut_age": load_tut_age_model,
    }

    args = all_args[os.path.splitext(os.path.basename(__file__))[0]]
    model = keras_load_model[args["app"]]

    session = tf.keras.backend.get_session()

    INPUT_NODE = model.inputs[0].op.name
    OUTPUT_NODE = model.outputs[0].op.name
    output_path = os.path.join(ROOT_DIR, "outputs/freeze")
    freeze_graph(session.graph, session, [out.op.name for out in model.outputs], output_path)
