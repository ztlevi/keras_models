import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K


# MAE
def mae_pred(y_true, y_pred):
    y_true = tf.sigmoid(y_true)
    levels = tf.cast(y_true > 0.5, tf.int8)
    true_labels = tf.cast(keras.backend.sum(levels, axis=1), tf.float32)

    predict_levels = tf.cast(y_pred > 0.5, tf.int8)
    predicted_labels = tf.cast(keras.backend.sum(predict_levels, axis=1), tf.float32)

    return K.mean(K.abs(true_labels - predicted_labels))


def task_importance_weights(label_array, num_classes):
    uniq = np.unique(label_array)
    num_examples = label_array.shape[0]

    m = np.zeros(num_classes-1)

    for i, t in enumerate(np.arange(np.min(uniq), np.max(uniq))):
        m_k = np.max([label_array[label_array > t].shape[0], num_examples - label_array[label_array > t].shape[0]])
        m[i] = np.sqrt(m_k)

    imp = m / np.max(m)
    return imp


class AgeDataGenerator(keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size, num_classes, image_shape=(224, 224)):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.n = len(image_filenames)

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        levels = [[1] * label + [0] * (self.num_classes - 1 - label) for label in batch_y]

        return (
            np.array(
                [cv2.resize(cv2.imread(file_name) / 255.0, self.image_shape) for file_name in batch_x]
            ),
            # np.stack(batch_y, np.array(levels))
            np.array(levels)
        )


class Linear_1_bias(keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Linear_1_bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.linear_1_bais = self.add_weight(name="Liner_1_Bais",
                                             shape=(self.num_classes - 1),
                                             initializer="zeros",
                                             trainable=True)
        # keras.backend.zeros(self.num_classes - 1)
        super(Linear_1_bias, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # return keras.backend.bias_add(x, self.linear_1_bais)
        x = x + self.linear_1_bais
        # probas = keras.backend.sigmoid(x)
        return x

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
        }
        base_config = super(Linear_1_bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)

def coral_loss(imp):
    def loss(levels, logits):
        val = -K.sum(
            (K.log(K.sigmoid(logits)) * levels + (K.log(K.sigmoid(logits)) - logits) * (1 - levels))
            * tf.convert_to_tensor(imp, dtype=tf.float32),
            axis=1,
            )
        return K.mean(val)

    return loss
