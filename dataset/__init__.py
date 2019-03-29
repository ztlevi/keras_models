import numpy as np
import cv2
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
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

        return (
            np.array(
                [cv2.resize(cv2.imread(file_name) / 255.0, self.image_shape) for file_name in batch_x]
            ),
            keras.utils.to_categorical(np.array(batch_y), num_classes=self.num_classes),
        )

