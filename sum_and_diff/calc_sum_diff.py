import pandas as pd
import numpy as np
from rtm import TsetlinMachine
import os
from time import strftime
from matplotlib import pyplot as plt

# GLOBAL PATHS:
TRAIN_FILE = os.path.join(os.getcwd(), "sum_and_diff/data/train copy.csv")
TEST_FILE = os.path.join(os.getcwd(), "sum_and_diff/data/test copy.csv")
PLT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "debug/run_"+strftime("%Y%m%d_%H%M%S")+"_err.png")

# NOTE: DEFINING A STDOUT LOGGER TO STORE ALL PRINT STATEMENTS FOR FUTURE USE
STDOUT_LOG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "run_"+strftime("%Y%m%d_%H%M%S")+".txt")

def read_data(file_path, train_file=False):
    data = pd.read_csv(file_path)
    idx = data['Index']
    X1 = data['X1']
    X2 = data['X2']
    if train_file:
        sum_y = data['Sum']
        diff_y = data['Difference']
        return idx, X1, X2, sum_y, diff_y
    else:
        return idx, X1, X2


def main():

    num_epochs = 30

    train_idx, train_X1, train_X2, sum_y, diff_y = read_data(TRAIN_FILE, train_file=True)
    test_idx, test_X1, test_X2 = read_data(TEST_FILE)

    rtm_sum = TsetlinMachine(number_of_clauses=100,
            number_of_features=10,
            number_of_states=200,
            s=2,
            threshold=50,
            max_target=10,
            min_target=0,
            logger=STDOUT_LOG)
    rtm_diff = TsetlinMachine(number_of_clauses=100,
            number_of_features=10,
            number_of_states=200,
            s=2,
            threshold=50,
            max_target=5,
            min_target=-5,
            logger=STDOUT_LOG)
    
    sum_err = []
    diff_err = []

    for epoch in range(num_epochs):
        bin_X = []
        for idx in range(len(train_idx)):
            bin_X1 = np.fromiter(np.binary_repr(train_X1[idx], 5), dtype=np.int32)
            bin_X2 = np.fromiter(np.binary_repr(train_X2[idx], 5), dtype=np.int32)
            bin_X.append(np.concatenate((bin_X1, bin_X2)))
        bin_X = np.array(bin_X)
        print("FITTING RTMs", file=open(STDOUT_LOG, "a"))   
        rtm_sum.fit(bin_X, sum_y, len(train_idx))
        rtm_diff.fit(bin_X, diff_y, len(train_idx))
        
        sum_err_epoch = 0
        diff_err_epoch = 0
        for idx in range(len(test_idx)):
            bin_X1 = np.fromiter(np.binary_repr(test_X1[idx], 5), dtype=np.int32)
            bin_X2 = np.fromiter(np.binary_repr(test_X2[idx], 5), dtype=np.int32)
            bin_X = np.concatenate((bin_X1, bin_X2))
            print("X1={}\tX2={}".format(test_X1[idx], test_X2[idx]), file=open(STDOUT_LOG, "a"))
            sum_X = rtm_sum.predict(bin_X)
            diff_X = rtm_diff.predict(bin_X)
            print("Sum: {}\t Diff: {}".format(sum_X, diff_X), file=open(STDOUT_LOG, "a"))
            sum_err_epoch += (test_X1[idx] + test_X2[idx] - sum_X)**2
            diff_err_epoch += (test_X1[idx]-test_X2[idx] - diff_X)**2
        sum_err.append(sum_err_epoch/len(test_idx))
        diff_err.append(diff_err_epoch/len(test_idx))
    print("Average Error (Summing RTM):{}\nAverage Error (Difference RTM):{}".format(np.mean(sum_err), np.mean(diff_err)))

    x = np.arange(num_epochs)
    plt.subplots()
    plt.plot(x, sum_err, label="error in sum")
    plt.plot(x, diff_err, label="error in difference")
    plt.suptitle("Testing Error")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.legend(loc="upper right")

    plt.savefig(PLT_PATH, bbox_inches="tight")
    plt.close()

    

if __name__=="__main__":
    main()