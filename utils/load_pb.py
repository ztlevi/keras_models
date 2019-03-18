from PIL import Image
import numpy as np
import tensorflow as tf

# https://i.imgur.com/tvOB18o.jpg
im = Image.open("/home/chichivica/Pictures/eagle.jpg").resize((299, 299), Image.BICUBIC)
im = np.array(im) / 255.0
im = im[None, ...]

graph_def = tf.GraphDef()

with tf.gfile.GFile("frozen_model.pb", "rb") as f:
    graph_def.ParseFromString(f.read())

graph = tf.Graph()

with graph.as_default():
    net_inp, net_out = tf.import_graph_def(
        graph_def, return_elements=["input_1", "predictions/Softmax"]
    )
    with tf.Session(graph=graph) as sess:
        out = sess.run(net_out.outputs[0], feed_dict={net_inp.outputs[0]: im})
        print(np.argmax(out))