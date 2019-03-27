import os
import pickle
import random
import sys

import numpy as np

import cv2
from definitions import AUDIENCE_DATASET_DIR


def validate_images(addrs):
    num_images = len(addrs)
    invalid_images_mask = np.ones((num_images,)).astype(bool)
    for i in range(num_images):
        if (i + 1) % 100 == 0 and i > 1:
            sys.stdout.write(f"\rLog: {i + 1}/{num_images} images processed...")
            sys.stdout.flush()
        addr = addrs[i]
        try:
            img = cv2.imread(addr)
        except Exception as e:
            invalid_images_mask[i] = False
            print(addr)
            print(str(e))
            return

    return invalid_images_mask


def dump_audience_pkl(output_path):
    addrs, age_labels, gender_labels, lines = [], [], [], []
    for j in range(4):
        with open(os.path.join(AUDIENCE_DATASET_DIR, "fold_{}_data.txt".format(j)), "r") as f:
            lines_fold = f.readlines()
            lines += lines_fold[1:]
    random.shuffle(lines)

    num_images = len(lines)
    for i, line in enumerate(lines):
        line_arr = line.split("\t")
        addr = "{}/faces/{}/coarse_tilt_aligned_face.{}.{}".format(
            AUDIENCE_DATASET_DIR, line_arr[0], line_arr[2], line_arr[1]
        )
        try:
            age = int(sum(list(eval(line_arr[3]))) / 2)
            gender = 0 if line_arr[4] == "f" else 1

            addrs += [addr]
            age_labels += [age]
            gender_labels += [gender]
        except TypeError as e:
            # resolve None age
            continue

    addrs = np.array(addrs)
    age_labels = np.array(age_labels)
    gender_labels = np.array(gender_labels)

    # Remove invalid images
    invalid_images_mask = validate_images(addrs)
    addrs = addrs[invalid_images_mask]
    age_labels = age_labels[invalid_images_mask]
    gender_labels = gender_labels[invalid_images_mask]

    data = {"addrs": addrs, "gender_labels": gender_labels, "age_labels": age_labels}
    pickle.dump(data, open(output_path, "wb"))
    print(f"\nFinish dumping data!!!")


def get_audience_dataset():
    output_path = os.path.join(AUDIENCE_DATASET_DIR, "audience.pkl")
    return pickle.load(open(output_path, "rb"))


if __name__ == "__main__":
    output_path = os.path.abspath(os.path.join(AUDIENCE_DATASET_DIR, "audience.pkl"))

    dump_audience_pkl(output_path)
