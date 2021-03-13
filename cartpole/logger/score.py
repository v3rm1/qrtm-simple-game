import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from collections import deque
import os
import csv
from time import strftime

AVG_SCORE = 195

# TODO: Change this when using on local. The current path is hardcoded for Peregrine.
# LAPTOP PATH
SCORES_DIR = os.path.join(os.getcwd(), "cartpole/logger/scores/")

# PEREGRINE PATH
# SCORES_DIR = "/data/s3893030/rtm_ql/logger/scores/"

SCORE_CSV = os.path.join(SCORES_DIR,
                         'score_' + strftime("%Y%m%d_%H%M%S") + ".csv")
SCORE_PNG = os.path.join(SCORES_DIR,
                         'score_' + strftime("%Y%m%d_%H%M%S") + ".png")


class ScoreLogger:
    """ """
    def __init__(self, env_name, mem_length=300):
        super().__init__()
        self.scores = deque(maxlen=mem_length)
        self.env_name = env_name

    def add_score(self, score, run, gamma, epsilon_decay_func, consecutive_runs=300, sedf_alpha=0, sedf_beta=0, sedf_delta=0, edf_epsilon_decay=0):
        """

        :param score: 
        :param run: 

        """
        self._save_csv(SCORE_CSV, score)
        self._save_png(input_scores=SCORE_CSV,
                       output_img=SCORE_PNG,
                       x_label="Runs",
                       y_label="Score",
                       avg_of_last=consecutive_runs,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True,
                       gamma=gamma,
                       epsilon_decay_func=epsilon_decay_func,
                       sedf_alpha=sedf_alpha,
                       sedf_beta=sedf_beta,
                       sedf_delta=sedf_delta,
                       edf_epsilon_decay=edf_epsilon_decay)
        self.scores.append(score)
        avg_score = np.mean(self.scores)
        # print("Scores:\nmin: {0}\tmax: {1}\tavg: {2}".format(np.min(self.scores), np.max(self.scores), np.mean(self.scores)))
        if avg_score >= AVG_SCORE and len(self.scores) >= consecutive_runs:
            solve_score = run - consecutive_runs
            print("Solved in {0} runs of {1} total runs".format(
                solve_score, run))
            self._save_csv(SCORE_CSV, score)
            self._save_png(input_scores=SCORE_CSV,
                           output_img=SCORE_PNG,
                           x_label="Trials",
                           y_label="Steps Taken",
                           avg_of_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False, 
                           gamma=gamma, 
                           epsilon_decay_func=epsilon_decay_func,
                           sedf_alpha=sedf_alpha,
                           sedf_beta=sedf_beta,
                           sedf_delta=sedf_delta,
                           edf_epsilon_decay=edf_epsilon_decay)
            exit()

    def _save_csv(self, path, score):
        """

        :param path: 
        :param score: 

        """
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])

    def _save_png(self, input_scores, output_img, x_label, y_label, avg_of_last, show_goal, show_trend, show_legend, gamma, epsilon_decay_func, sedf_alpha, sedf_beta, sedf_delta, edf_epsilon_decay):
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
        x = []
        y = []
        with open(input_scores, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="Score VS Run")
        avg_range = avg_of_last if avg_of_last is not None else len(x)
        plt.plot(x[-avg_range:],
                 [np.mean(y[-avg_range:])] * len(y[-avg_range:]),
                 linestyle="--",
                 label="Avg over last " + str(avg_range) + " runs")
        if show_goal:
            plt.plot(x, [AVG_SCORE] * len(x),
                     linestyle=":",
                     label="Goal: " + str(AVG_SCORE))
        if show_trend and len(x) > 1:
            trend_x = x[1:]
            curve_fn = np.poly1d(
                np.polyfit(np.array(trend_x), np.array(y[1:]), 1))
            plt.plot(trend_x,
                     curve_fn(trend_x),
                     linestyle="-.",
                     label="Trend line")

        plt.suptitle(self.env_name + " Gamma: " + str(gamma) + " Epsilon Decay Function: " + str(epsilon_decay_func))
        if epsilon_decay_func == "SEDF":
            title_str = "alpha: " + str(sedf_alpha) + "  beta: " + str(sedf_beta) + "  delta: " + str(sedf_delta)
        else:
            title_str = "Epsilon Decay: " + str(edf_epsilon_decay)
        plt.title(title_str)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_img, bbox_inches="tight")
        plt.close()


#TODO: Add more plots - epsilon for each run
