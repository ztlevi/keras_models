from math import ceil
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from skimage.io import imread

from dataset.Audience import get_audience_dataset

data = get_audience_dataset()
addrs = data["addrs"]
gender_labels = data["gender_labels"]
age_labels = data["age_labels"]

num_images = len(addrs)

# Plotting Age distribution
sorted_age = sorted(age_labels)
fit = norm.pdf(sorted_age, np.mean(sorted_age), np.std(sorted_age))
plt.figure("Age distribution")
plt.plot(sorted_age, fit, "-o")
plt.hist(sorted_age, bins=20, density=True)
plt.show()


batch_size = 64
num_class = 2

# create list of batches to shuffle the data
batches_list = list(range(int(ceil(float(num_images) / batch_size))))
shuffle(batches_list)

class_names = ["female", "male"]

# loop over batches
for x in range(5):
    i = batches_list[x]

    i_s = i * batch_size  # index of the first image in this batch
    i_e = min([(i + 1) * batch_size, num_images])  # index of the last image in this batch

    # read batch images and remove training mean
    images = [imread(addr) for addr in addrs[i_s:i_e, ...]]

    # read labels and convert to one hot encoding
    seg_gender = gender_labels[i_s:i_e]
    seg_age = age_labels[i_s:i_e]

    plt.figure(figsize=(10, 10))
    for j in range(25):
        plt.subplot(5, 5, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if j < len(images):
            plt.imshow(images[j], cmap=plt.cm.binary)
            plt.xlabel(class_names[seg_gender[j]] + "_" + str(seg_age[j]))

plt.show()
