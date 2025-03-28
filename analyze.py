import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md
from paprika import utils


def generate_histograms(dynamic_restraints, window_list, data_dir_names):

    for dynamic_restraint in dynamic_restraints:
        fig = plt.figure(figsize=(16, 9))
        for window in window_list:
            