import multiprocessing
import os

####################### PROJECT DIR #######################
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def get_relative_path(file):
    return os.path.abspath(file)[len(ROOT_DIR) + 1 :]


####################### SETTING #######################
NUM_CPUS = multiprocessing.cpu_count()

####################### DATASET #######################
IMDB_WIKI_DATASET_DIR = ["/media/ztlevi/HDD/IMDB-WIKI", "/data/workspace_tingzhou/IMDB-WIKI/"]
AUDIENCE_DATASET_DIR = "/media/ztlevi/HDD/AdienceBenchmarkOfUnfilteredFaces/"
UTKFace_DATASET_DIR = [
    "/media/ztlevi/HDD/UTKFace/UTKFace/",
    "/data/workspace_tingzhou/UTKFace/UTKFace/",
]
AFFECTNET_DATASET_DIR = "/media/ztlevi/HDD/Affectnet/"
HANDTIP_DATASET_DIR = "/media/ztlevi/HDD/HandTip/Train_t/"

####################### running arguments #######################
all_args = {
    "training/age/age_mobilenet_v1_fine_tuning_imdb_wiki.py": {
        "use_remote": False,
        "GPUS": "4,5,6,7",
    },
    "training/age/age_mobilenet_v1_fine_tuning_utkface.py": {
        "use_remote": False,
        "GPUS": "4,5,6,7",
    },
    "training/age/age_mobilenet_v1_fine_tuning_audience.py": {
        "use_remote": False,
        "GPUS": "4,5,6,7",
    },
    "training/7expr/7expr_mobilenet_v1_train_affectnet.py": {"use_remote": False},
    "utils/freeze_model.py": {"app": "general"},
}
