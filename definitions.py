import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
IMDB_WIKI_DATASET_DIR = ["/media/ztlevi/HDD/IMDB-WIKI", "/data/workspace_tingzhou/IMDB-WIKI"]
AUDIENCE_DATASET_DIR = "/media/ztlevi/HDD/AdienceBenchmarkOfUnfilteredFaces"
UTKFace_DATASET_DIR = [
    "/media/ztlevi/HDD/UTKFace/UTKFace/",
    "/data/workspace_tingzhou/UTKFace/UTKFace/",
]

all_args = {
    "age_mobilenet_v1_fine_tuning_imdb_wiki": {"use_remote": False, "GPUS": "4,5,6,7"},
    "age_mobilenet_v1_fine_tuning_utkface": {"use_remote": False, "GPUS": "4,5,6,7"},
    "age_mobilenet_v1_fine_tuning_audience": {"use_remote": False, "GPUS": "4,5,6,7"},
}
