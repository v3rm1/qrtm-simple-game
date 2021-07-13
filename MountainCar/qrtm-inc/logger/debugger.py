import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from collections import deque
import os
import csv
from time import strftime

plt.style.use('seaborn-muted')

# TODO: Change this when using on local. The current path is hardcoded for Peregrine.
# LAPTOP PATH
DEBUG_PLOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "debug/")

# PEREGRINE PATH
# DEBUG_PLOT_DIR = "/data/s3893030/cartpole/logger/debug/"

TD_ERROR_PLOT = os.path.join(DEBUG_PLOT_DIR,
                         'RMS_TD_Err' + strftime("%Y%m%d_%H%M%S") + ".png")


class DebugLogger:
    """ """
    def __init__(self, env_name):
        super().__init__()
        self.env_name = env_name

    def add_watcher(self, td_error, n_clauses, T, feature_length):
        """

        :param score: 
        :param run: 

        """
        self._save_png(td_error=td_error,
                       n_runs=len(td_error),
                       output_img=TD_ERROR_PLOT,
                       x_label="Episodes",
                       y_label="RMS TD Error for episode",
                       show_legend=True,
                       n_clauses=n_clauses,
                       T=T,
                       feature_length=feature_length
                       )
        return

    def _save_png(self, td_error, n_runs, output_img, x_label, y_label, show_legend, n_clauses, T, feature_length):
        """

        :param input_scores: 
        :param output_img: 
        :param x_label: 
        :param y_label: 
        :param avg_of_last: 
        :param show_goal: 
        :param show_trend: 
        :param show_legend: 

        """
        x = np.arange(n_runs)
        plt.subplots()
        plt.plot(x, td_error, label="RMS TD Err")
        plt.suptitle(self.env_name + ": RMS TD Error over " + str(n_runs) + " runs")
        plt.title("n_clauses: " + str(n_clauses) + " T: " + str(T) + " bits_per_feature: " + str(feature_length))
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper right")

        plt.savefig(output_img, bbox_inches="tight")
        plt.close()
