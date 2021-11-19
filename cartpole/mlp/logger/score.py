import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from collections import deque
import os
import csv
from time import strftime

AVG_SCORE = 200
CONSECUTIVE_RUNS = 3000

SCORES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scores/")
SCORE_CSV = os.path.join(SCORES_DIR,
                         'score_' + strftime("%Y%m%d_%H%M%S") + ".csv")
SCORE_PNG = os.path.join(SCORES_DIR,
                         'score_' + strftime("%Y%m%d_%H%M%S") + ".png")


class ScoreLogger:
    """ """
    def __init__(self, env_name):
        super().__init__()
        self.scores = deque(maxlen=CONSECUTIVE_RUNS)
        self.env_name = env_name

    def add_score(self, score, run):
        """

        :param score: 
        :param run: 

        """
        self._save_csv(SCORE_CSV, score)
        self._save_png(input_scores=SCORE_CSV,
                       output_img=SCORE_PNG,
                       x_label="Episode",
                       y_label="Steps Taken",
                       avg_of_last=CONSECUTIVE_RUNS,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True)
        self.scores.append(score)
        avg_score = np.mean(self.scores)
        print("Scores:\nmin: {0}\tmax: {1}\tavg: {2}".format(
            np.min(self.scores), np.max(self.scores), np.mean(self.scores)))
        if avg_score >= AVG_SCORE and len(self.scores) >= CONSECUTIVE_RUNS:
            solve_score = run - CONSECUTIVE_RUNS
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
                           show_legend=False)
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

    def _save_png(self, input_scores, output_img, x_label, y_label,
                  avg_of_last, show_goal, show_trend, show_legend):
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
                y.append(float(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="Steps Taken VS Episode")
        avg_range = avg_of_last if avg_of_last is not None else len(x)
        plt.plot(x[-avg_range:],
                 [np.mean(y[-avg_range:])] * len(y[-avg_range:]),
                 linestyle="--",
                 label="Avg over last" + str(avg_range) + "runs")
        if show_goal:
            plt.plot(x, [AVG_SCORE] * len(x),
                     linestyle=":",
                     label="Limit: " + str(AVG_SCORE))
        if show_trend and len(x) > 1:
            trend_x = x[1:]
            curve_fn = np.poly1d(
                np.polyfit(np.array(trend_x), np.array(y[1:]), 1))
            plt.plot(trend_x,
                     curve_fn(trend_x),
                     linestyle="-.",
                     label="Trend line")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_img, bbox_inches="tight")
        plt.close()

#TODO: Add more plots - epsilon for each run 