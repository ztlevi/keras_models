# https://stackoverflow.com/questions/51858203/cant-import-frozen-graph-with-batchnorm-layer?noredirect=1&lq=1
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io

from definitions import ROOT_DIR
from training.age import mae_pred


def freeze_graph(graph, session, output, output_path):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output
        )
        graph_io.write_graph(graphdef_frozen, output_path, "frozen_model.pb", as_text=False)


keras.backend.set_learning_phase(0)  # this line most important

checkpoint_path = (
    "/home/ztlevi/Developer/keras_models/outputs/checkpoints/age_mobilenet_v1_utkface/ckpt.h5"
)
model = keras.models.load_model(checkpoint_path, custom_objects={"mae_pred": mae_pred})

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
