import os
import pickle

import numpy as np
import pandas as pd

from definitions import HANDTIP_DATASET_DIR


def dump_handtip_pkl(output_path):
    csv_data = pd.read_csv(os.path.join(HANDTIP_DATASET_DIR, "Train.csv"))
    data = {}
    for column in csv_data.columns:
        data[column] = csv_data[column].values

    shuffle_mask = np.arange(len(data["filename"]))
    np.random.shuffle(shuffle_mask)
    for column in data:
        data[column] = data[column][shuffle_mask]

    data["filename"] = HANDTIP_DATASET_DIR + data["filename"]
    pickle.dump(data, open(output_path, "wb"))
    print(f"\nFinish dumping data!!!")


def get_handtip_dataset():
    output_path = os.path.join(HANDTIP_DATASET_DIR, "..", "handtip.pkl")
    return pickle.load(open(output_path, "rb"))


if __name__ == "__main__":
    output_path = os.path.abspath(os.path.join(HANDTIP_DATASET_DIR, "..", "handtip.pkl"))

    dump_handtip_pkl(output_path)
