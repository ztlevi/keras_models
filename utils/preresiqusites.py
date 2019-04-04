import os

from definitions import ROOT_DIR


def create_output_dirs_if_not_exists():
    dirs = [
        "outputs",
        "outputs/checkpoints",
        "outputs/best_checkpoints",
        "outputs/logs",
        "outputs/freeze",
        "outputs/figures"
    ]

    for directory in dirs:
        path = os.path.join(ROOT_DIR, directory)
        if not os.path.exists(path):
            os.makedirs(path)


def run_preresiqusites():
    create_output_dirs_if_not_exists()
