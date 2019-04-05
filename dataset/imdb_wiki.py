import os
import pickle
import sys
from datetime import date

import numpy as np
import scipy.io

import cv2
from definitions import IMDB_WIKI_DATASET_DIR


def read_mat_data(name):
    mat = scipy.io.loadmat(os.path.join(IMDB_WIKI_DATASET_DIR, name + "_crop", name + ".mat"))
    mat = mat[name][0][0]
    data = {}
    for k in mat.dtype.names:
        data[k] = mat[k].flatten()

    return data


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


def dump_imdb_wiki_pkl(output_path):
    ids = ["imdb", "wiki"]

    all_gender_labels = np.array([])
    all_age_labels = np.array([])
    all_addrs = np.array([])
    for id in ids:
        data = read_mat_data(id)

        addrs = data["full_path"]
        gender_labels = data["gender"]
        photo_taken = data["photo_taken"]
        dob = data["dob"]

        num_id_images = len(data["full_path"])
        age_labels = []

        age_invalid_mask = np.ones(num_id_images).astype(bool)
        for photo_idx in range(num_id_images):
            try:
                # The age of a person can be calculated based on the date of birth and the time when the photo was taken (note that we assume that the photo was taken in the middle of the year):
                age = date.fromordinal(
                    date.toordinal(date(photo_taken[photo_idx], 7, 1)) - dob[photo_idx]
                )
                if age.year > 100 or age == 0:
                    raise Exception("Age is invalid...")
                age_labels += [age.year]
            except:
                age_labels += [0]
                # Cases like photo taken year is lower than date of birth
                age_invalid_mask[photo_idx] = [False]
        age_labels = np.array(age_labels)

        # Clean up NAN gender
        gender_invalid_mask = [
            False if np.isnan(gender_labels[i]) else True for i in range(len(gender_labels))
        ]

        mask = gender_invalid_mask & age_invalid_mask
        gender_labels = gender_labels[mask]
        age_labels = age_labels[mask]
        addrs = addrs[mask]

        # raise addrs' items
        for i in range(len(addrs)):
            addrs[i] = addrs[i][0]

        addrs = IMDB_WIKI_DATASET_DIR + "/" + (id + "_crop/") + addrs
        all_gender_labels = np.append(all_gender_labels, gender_labels)
        all_age_labels = np.append(all_age_labels, age_labels)
        all_addrs = np.append(all_addrs, addrs)

    # Remove invalid images, indexes of the invalid images are generated
    # during `storage_add_images`. For imdb-wiki dataset, it means the image
    # is all black
    invalid_images_mask = validate_images(all_addrs)
    all_addrs = all_addrs[invalid_images_mask]
    all_gender_labels = all_gender_labels[invalid_images_mask]
    all_gender_labels = all_gender_labels.astype(int)
    all_age_labels = all_age_labels[invalid_images_mask]
    all_age_labels = all_age_labels.astype(int)

    # Shuffle
    shuffle_mask = np.arange(len(all_addrs))
    np.random.shuffle(shuffle_mask)
    all_addrs = all_addrs[shuffle_mask]
    all_age_labels = all_age_labels[shuffle_mask]
    all_gender_labels = all_gender_labels[shuffle_mask]

    data = {"addrs": all_addrs, "gender_labels": all_gender_labels, "age_labels": all_age_labels}
    pickle.dump(data, open(output_path, "wb"))
    print(f"\nFinish dumping data!!!")


def get_imdb_wiki_dataset(use_remote=False):
    output_path = os.path.join(IMDB_WIKI_DATASET_DIR[use_remote], "imdb-wiki.pkl")
    return pickle.load(open(output_path, "rb"))


if __name__ == "__main__":
    output_path = os.path.join(IMDB_WIKI_DATASET_DIR[0], "imdb-wiki.pkl")
    dump_imdb_wiki_pkl(output_path)
