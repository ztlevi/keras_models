import os
import pickle
import sys

import numpy as np
import pandas as pd
import tables

import cv2
from definitions import AFFECTNET_DATASET_DIR

class_names = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger",
    "Contempt",
    "None",
    "Uncertain",
    "Non-Face",
]
class_names_to_ids = dict(zip(class_names, range(len(class_names))))


def get_csv_data(dataset_dir, is_training):
    """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
    """
    if is_training:
        manually_training_csv = os.path.join(
            dataset_dir, "Manually_Annotated_file_lists", "training.csv"
        )
        auto_training_csv = os.path.join(
            dataset_dir, "Automatically_Annotated_file_lists", "automatically_annotated.csv"
        )
        # arr = []
        manually_csv_data = pd.read_csv(manually_training_csv, index_col=None, header=0)
        manually_csv_data["subDirectory_filePath"] = [
            os.path.join("Manually_Annotated_Images", sub)
            for sub in manually_csv_data["subDirectory_filePath"]
        ]
        # arr.append(manually_csv_data)
        print("Finish loading manually_csv...")

        # auto_csv_data = pd.read_csv(auto_training_csv, index_col=None, header=0)
        # auto_csv_data["subDirectory_filePath"] = [
        #     os.path.join("Automatically_Annotated_Images", sub)
        #     for sub in auto_csv_data["subDirectory_filePath"]
        # ]
        # arr.append(auto_csv_data)
        # print("Finish loading auto_csv...")
        #
        # csv_data = pd.concat(arr, axis=0, ignore_index=True)
        # print("Finish concating manually_csv and auto_csv...")
        csv_data = manually_csv_data
    else:
        csv_data = pd.read_csv(
            os.path.join(dataset_dir, "Manually_Annotated_file_lists", "validation.csv")
        )
        csv_data["subDirectory_filePath"] = [
            os.path.join("Manually_Annotated_Images", sub)
            for sub in csv_data["subDirectory_filePath"]
        ]
        print("Finish loading validation_csv...")

    # Shuffle csv data
    csv_data = csv_data.sample(frac=1).reset_index(drop=True)
    # return csv_data[:5500]
    return csv_data


def validate_images(addrs):
    num_images = len(addrs)
    invalid_images_mask = np.ones((num_images,)).astype(bool)
    for i in range(num_images):
        if (i + 1) % 100 == 0 and i > 1:
            sys.stdout.write(f"\rLog: {i + 1}/{num_images} images processed...")
            sys.stdout.flush()

        # read an image and resize to img_shape
        # cv2 load images as BGR, convert it to RGB
        addr = addrs[i]

        if not os.path.exists(addr):
            invalid_images_mask[i] = False
            continue
        try:
            img = cv2.imread(addr)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            invalid_images_mask[i] = False
            print(addr)
            print(str(e))
            continue

        # Remove invalid images, indexes of the invalid images are generated
        # during `storage_add_images`. For imdb-wiki dataset, it means the image
        # is all black
        if cv2.countNonZero(gray_img) == 0:
            invalid_images_mask[i] = False
            continue

    return invalid_images_mask


def dump_affectnet_pkl(output_path):
    training_csv_data = get_csv_data(AFFECTNET_DATASET_DIR, True)
    validation_csv_data = get_csv_data(AFFECTNET_DATASET_DIR, False)

    data = {}
    for key, csv_data in {"training": training_csv_data, "validation": validation_csv_data}.items():
        addrs = AFFECTNET_DATASET_DIR + "/" + csv_data.subDirectory_filePath.values

        invalid_images_mask = validate_images(addrs)
        addrs = addrs[invalid_images_mask]
        valence_labels = csv_data.valence.values[invalid_images_mask]
        arousal_labels = csv_data.arousal.values[invalid_images_mask]
        expression_labels = csv_data.expression.values[invalid_images_mask]

        shuffle_mask = np.arange(len(addrs))
        np.random.shuffle(shuffle_mask)
        addrs = addrs[shuffle_mask]
        valence_labels = valence_labels[shuffle_mask]
        arousal_labels = arousal_labels[shuffle_mask]
        expression_labels = expression_labels[shuffle_mask]

        data[key] = {
            "addrs": addrs,
            "valence_labels": valence_labels,
            "arousal_labels": arousal_labels,
            "expression_labels": expression_labels,
        }
    pickle.dump(data, open(output_path, "wb"))
    print(f"\nFinish dumping data!!!")


def get_affectnet_dataset():
    output_path = os.path.join(AFFECTNET_DATASET_DIR, "affectnet.pkl")
    return pickle.load(open(output_path, "rb"))


if __name__ == "__main__":
    output_path = os.path.join(AFFECTNET_DATASET_DIR, "affectnet.pkl")
    dump_affectnet_pkl(output_path)
