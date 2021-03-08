import pandas as pd
import numpy as np
from rtm import TsetlinMachine
import os
from time import strftime

# GLOBAL PATHS:
TRAIN_FILE = os.path.join(os.getcwd(), "sum_and_diff/data/train.csv")
TEST_FILE = os.path.join(os.getcwd(), "sum_and_diff/data/test.csv")

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
    train_idx, train_X1, train_X2, sum_y, diff_y = read_data(TRAIN_FILE, train_file=True)
    test_idx, test_X1, test_X2 = read_data(TEST_FILE)

    rtm_sum = TsetlinMachine(number_of_clauses=20,
            number_of_features=10,
            number_of_states=200,
            s=2,
            threshold=100,
            max_target=50,
            min_target=-50,
            logger=STDOUT_LOG)
    rtm_diff = TsetlinMachine(number_of_clauses=20,
            number_of_features=10,
            number_of_states=200,
            s=2,
            threshold=100,
            max_target=50,
            min_target=-50,
            logger=STDOUT_LOG)
    bin_X = []
    for idx in range(len(train_idx)):
        bin_X1 = np.fromiter(np.binary_repr(train_X1[idx], 5), dtype=np.int32)
        bin_X2 = np.fromiter(np.binary_repr(train_X2[idx], 5), dtype=np.int32)
        bin_X.append(np.concatenate((bin_X1, bin_X2)))
    bin_X = np.array(bin_X)
    print("FITTING RTMs", file=open(STDOUT_LOG, "a"))   
    rtm_sum.fit(bin_X, sum_y, len(train_idx))
    rtm_diff.fit(bin_X, diff_y, len(train_idx))
    
    sum_err = []
    diff_err = []
    
    for idx in range(len(test_idx)):
        bin_X1 = np.fromiter(np.binary_repr(test_X1[idx], 5), dtype=np.int32)
        bin_X2 = np.fromiter(np.binary_repr(test_X2[idx], 5), dtype=np.int32)
        bin_X = np.concatenate((bin_X1, bin_X2))
        print("X1={}\tX2={}".format(test_X1[idx], test_X2[idx]), file=open(STDOUT_LOG, "a"))
        sum_X = rtm_sum.predict(bin_X)
        diff_X = rtm_diff.predict(bin_X)
        print("Sum: {}\t Diff: {}".format(sum_X, diff_X), file=open(STDOUT_LOG, "a"))
        sum_err.append(test_X1[idx] + test_X2[idx] - sum_X)
        diff_err.append(test_X1[idx]-test_X2[idx] - diff_X)
    print("Average Error (Summing RTM):{}\nAverage Error (Difference RTM):{}".format(np.mean(sum_err), np.mean(diff_err)), file=open(STDOUT_LOG, "a"))



    

if __name__=="__main__":
    main()