import glob


def get_latest_checkpoint(checkpoint_dir):
    dirs = glob.glob(checkpoint_dir + "/*")
    if not dirs:
        return ""
    return sorted(dirs)[-1]
