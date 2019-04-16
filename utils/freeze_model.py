# https://stackoverflow.com/questions/51858203/cant-import-frozen-graph-with-batchnorm-layer?noredirect=1&lq=1
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io

from definitions import ROOT_DIR
from training.age import Linear_1_bias, coral_loss, mae_pred


def freeze_graph(graph, session, output, output_path):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output
        )
        graph_io.write_graph(graphdef_frozen, output_path, "frozen_model.pb", as_text=False)


keras.backend.set_learning_phase(0)  # this line most important

model = keras.models.load_model(
    os.path.join(ROOT_DIR, "outputs", "checkpoints", "7expr_mobilenet_v1_affectnet", "ckpt.h5")
)

# # Age model
# loss = coral_loss(np.ones(101 - 1))
# checkpoint_path = (
#     "/home/ztlevi/Developer/keras_models/outputs/checkpoints/age_mobilenet_v1_utkface/ckpt.h5"
# )
# model = keras.models.load_model(
#     checkpoint_path,
#     custom_objects={"loss": loss, "mae_pred": mae_pred, "Linear_1_bias": Linear_1_bias},
# )

# TUT age model
# model = keras.models.load_model("/home/ztlevi/Developer/keras_models/outputs/tut/age/model.h5",
#                                     custom_objects={'relu6': keras.layers.ReLU(6.),
#                                                     'GlorotUniform': keras.initializers.glorot_uniform(),
#                                                     'DepthwiseConv2D': keras.layers.DepthwiseConv2D})

session = tf.keras.backend.get_session()

INPUT_NODE = model.inputs[0].op.name
OUTPUT_NODE = model.outputs[0].op.name
output_path = os.path.join(ROOT_DIR, "outputs/freeze")
freeze_graph(session.graph, session, [out.op.name for out in model.outputs], output_path)
