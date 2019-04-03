import glob
import numpy as np
import os
import pickle
import random
import sys

import tables

import cv2
from definitions import UTKFace_DATASET_DIR


def storage_add_images(img_storage, age_storage, gender_storage, race_storage, addrs, img_shape):
    num_images = len(addrs)
    invalid_addrs = []
    for i, addr in enumerate(addrs):
        if (i + 1) % 100 == 0 and i > 1:
            sys.stdout.write(f"\rLog: {i + 1}/{num_images} images processed...")
            sys.stdout.flush()

        try:
            img = cv2.imread(addr)
        except Exception as e:
            print(addr)
            print(str(e))
            continue

        img = cv2.resize(img, img_shape, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_storage.append(img[None])

        # addr name formatted as [age]_[gender]_[race]_[data&time]
        filename = os.path.basename(addr)
        info = filename.split("_")
        try:
            age_storage.append([int(info[0])])
            gender_storage.append([int(info[1])])
            race_storage.append([int(info[2])])
        except Exception as e:
            invalid_addrs += [addr]
            print(str(e))
    print(f"\nFinish converting data!!!")
    print(invalid_addrs)


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


def get_utkface_dataset():
    output_path = os.path.join(UTKFace_DATASET_DIR, "..", "UTKFace.pkl")
    return pickle.load(open(output_path, "rb"))


if __name__ == "__main__":
    output_path = os.path.abspath(os.path.join(UTKFace_DATASET_DIR, "..", "UTKFace.pkl"))

    dump_utkface_pkl(output_path)
