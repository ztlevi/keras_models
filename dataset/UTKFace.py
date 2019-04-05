import glob
import os
import pickle
import random
import sys

import numpy as np
import tables

import cv2
from definitions import UTKFace_DATASET_DIR


def dump_utkface_pkl(output_path):
    addrs = glob.glob(UTKFace_DATASET_DIR + "*.jpg")
    random.shuffle(addrs)
    print(len(addrs))

    addrs_final, age_labels, gender_labels, race_labels = [], [], [], []
    for addr in addrs:
        filename = os.path.basename(addr)
        info = filename.split("_")
        # Some image file has missing property
        if len(info) < 4:
            continue
        try:
            age = int(info[0])
            gender = int(info[1])
            race = int(info[2])
        except Exception as e:
            print(e)
            continue
        if age > 100:
            age = 100
        addrs_final += [addr]
        age_labels += [age]
        gender_labels += [gender]
        race_labels += [race]

    data = {
        "addrs": np.array(addrs_final),
        "gender_labels": np.array(gender_labels),
        "age_labels": np.array(age_labels),
        "race_labels": np.array(race_labels),
    }
    pickle.dump(data, open(output_path, "wb"))
    print(f"\nFinish dumping data!!!")


def get_utkface_dataset(use_remote=False):
    output_path = os.path.join(UTKFace_DATASET_DIR[use_remote], "..", "UTKFace.pkl")
    return pickle.load(open(output_path, "rb"))


if __name__ == "__main__":
    output_path = os.path.abspath(os.path.join(UTKFace_DATASET_DIR[0], "..", "UTKFace.pkl"))

    dump_utkface_pkl(output_path)
