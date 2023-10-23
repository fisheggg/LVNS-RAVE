import os
import glob
import numpy as np
import seaborn as sns


def plot_history(dir, name):
    files = sorted(glob.glob(os.path.join(dir, f"*/*{name}*.npy")))