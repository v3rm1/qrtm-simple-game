import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from collections import deque
import os
import csv
from time import strftime

# TODO: Change this when using on local. The current path is hardcoded for Peregrine.
# LAPTOP PATH
DEBUG_PLOT_DIR = os.path.join(os.getcwd(), "PER_test_simple/logger/debug/")

# PEREGRINE PATH
# DEBUG_PLOT_DIR = "/data/s3893030/rtm_ql_PER/logger/debug/"

Q_VALS_PNG = os.path.join(DEBUG_PLOT_DIR,
                         'avg_q_vals_' + strftime("%Y%m%d_%H%M%S") + ".png")
AVG_TD_ERR_PNG = os.path.join(DEBUG_PLOT_DIR,
                         'avg_td_err_' + strftime("%Y%m%d_%H%M%S") + ".png")

class DebugLogger:
    """ """
    def __init__(self, env_name):
        super().__init__()
        self.env_name = env_name

    def add_watcher(self, q_list_0, q_list_1, q_list_total, n_clauses, T, feature_length, error_list):
        """

        :param score: 
        :param run: 

        """
        self._save_png(q_0=q_list_0,
                       q_1=q_list_1,
                       q_total=q_list_total,
                       n_runs=len(q_list_0),
                       output_img=Q_VALS_PNG,
                       x_label="Runs",
                       y_label="Avg Q_Value",
                       show_legend=True,
                       n_clauses=n_clauses,
                       T=T,
                       feature_length=feature_length
                       )
        self._plot_error(error_list=error_list,
                         n_runs=len(q_list_0),
                         x_label="Runs",
                         y_label="TD Error: Avg over Run",
                         output_img=AVG_TD_ERR_PNG)
        return

    def _save_png(self, q_0, q_1, q_total, n_runs, output_img, x_label, y_label, show_legend, n_clauses, T, feature_length):
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
        plt.plot(x, q_0, label="q_0")
        plt.plot(x, q_1, label="q_1")
        plt.plot(x, q_total, 'r--', label="q_total")
        plt.suptitle(self.env_name + " : Avg Q-values over " + str(n_runs) + " runs")
        plt.title("n_clauses: " + str(n_clauses) + " T: " + str(T) + " bits_per_feature: " + str(feature_length))
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_img, bbox_inches="tight")
        plt.close()

    def _plot_error(self, error_list, n_runs, x_label, y_label, output_img, show_legend=False):
        x = np.arange(n_runs)
        plt.subplots()
        plt.plot(x, error_list, label="avg_td_err")
        plt.suptitle(self.env_name + " : Avg TD-Error over " + str(n_runs) + " runs")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_img, bbox_inches="tight")
        plt.close()
