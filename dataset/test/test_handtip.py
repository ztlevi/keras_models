from math import ceil
from random import shuffle

import matplotlib.patches as patches
import matplotlib.pyplot as plt

import cv2
from dataset.HandTip import get_handtip_dataset

data = get_handtip_dataset()
addrs = data["filename"]
labels = data["class"]
xmin = data["xmin"]
ymin = data["ymin"]
xmax = data["xmax"]
ymax = data["ymax"]

num_images = len(addrs)

batch_size = 64
num_class = 2

# create list of batches to shuffle the data
batches_list = list(range(int(ceil(float(num_images) / batch_size))))
shuffle(batches_list)

# loop over batches
for x in range(5):
    i = batches_list[x]

    i_s = i * batch_size  # index of the first image in this batRh
    i_e = min([(i + 1) * batch_size, num_images])  # index of the last image in this batch

    # read batch images and remove training mean
    images = [cv2.cvtColor(cv2.imread(addr), cv2.COLOR_BGR2RGB) for addr in addrs[i_s:i_e, ...]]
    seg_xmin = xmin[i_s:i_e, ...]
    seg_ymin = ymin[i_s:i_e, ...]
    seg_xmax = xmax[i_s:i_e, ...]
    seg_ymax = ymax[i_s:i_e, ...]

    # read labels and convert to one hot encoding

    fig = plt.figure(figsize=(10, 10))
    for j in range(25):
        ax = fig.add_subplot(5, 5, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if j < len(images):
            plt.imshow(images[j], cmap=plt.cm.binary)
            rect = patches.Rectangle(
                (seg_xmin[j], seg_ymin[j]),
                seg_xmax[j] - seg_xmin[j],
                seg_ymax[j] - seg_ymin[j],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            # plt.xlabel(class_names[seg_gender[j]] + "_" + str(seg_age[j]))

plt.show()
