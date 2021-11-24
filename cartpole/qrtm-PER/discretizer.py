import numpy as np
import gym
import os
import pandas as pd
import matplotlib.pyplot as plt

class CustomDiscretizer:
    def __init__(self):
        super().__init__()
        self.bin_labels = []



    def _binned_binarizer(self, fp_num, range_min, range_max, n_bins=15):
        binary_rep = np.zeros(shape=n_bins+1, dtype=int)
        bin_delta = (np.absolute(range_max) + np.absolute(range_min))/n_bins
        if fp_num < 0:
            binary_rep[0] = 1
        else:
            binary_rep[0] = 0
        for i in range(1, n_bins+1):
            bin_min = range_min + (i-1) * bin_delta
            bin_max = range_min + (i) * bin_delta
            if bin_min <= np.absolute(fp_num) <= bin_max:
                binary_rep[i] = 1
        return binary_rep

    def _unsigned_binarizer(self, fp_num, range_min, range_max, n_bins=16):
        binary_rep = np.zeros(shape=n_bins, dtype=int)
        bin_delta = (np.absolute(range_max) + np.absolute(range_min))/n_bins
        for i in range(0, n_bins):
            bin_min = range_min + (i) * bin_delta
            bin_max = range_min + (i+1) * bin_delta
            if bin_min <= fp_num <= bin_max:
                binary_rep[i] = 1
                bin_label = "bin"+str(i)
        self.bin_labels.append(bin_label)
        return binary_rep

    def _greater_than_binarizer(self, fp_num, n_places=15):
        binary_rep = np.zeros(shape=n_places+1, dtype=int)
        if fp_num < 0:
            binary_rep[0] = 1
        else:
            binary_rep[0] = 0
        for i in range(1, n_places+1):
            if np.absolute(fp_num) > (i-1):
                binary_rep[n_places-i+1] = 1
                bin_label = "bin"+str(i-1)
        self.bin_labels.append(bin_label)
        return binary_rep

    def _less_than_binned_binarizer(self, fp_num, range_min, range_max, n_bins=16):
        if fp_num > range_max:
            fp_num = range_max
            
        binary_rep = np.zeros(shape=n_bins, dtype=int)
        bin_delta = (np.absolute(range_max) + np.absolute(range_min))/n_bins
        for i in range(1, n_bins+1):
            bin_max = range_min + i * bin_delta
            if fp_num <= bin_max:
                binary_rep[i-1] = 1
                bin_label = "bin"+str(i)

        self.bin_labels.append(bin_label)
        return binary_rep


    def _simple_binarizer(self, fp_num, bits):
        if fp_num < 0:
            return np.fromiter(np.binary_repr(3, width=bits+1), int)
        elif fp_num > 0:
            return np.fromiter(np.binary_repr(1, width=bits+1), int)
        else:
            return np.fromiter(np.binary_repr(0, width=bits+1), int)

    def _quartile_binner(self, fp_num, range_max, shape=4):
        binary_rep = np.zeros(shape=shape, dtype=int)
        if fp_num < 0:
            binary_rep[0] = 1
        else:
            binary_rep[0] = 0
        for i in range(0, 4):
            if (i*0.25*range_max) <= np.absolute(fp_num) < ((i+1)*0.25*range_max):
                # The floating point value belongs to the i+1th quartile.
                binary_rep[1:3] = np.fromiter(np.binary_repr(i+1), int)
            elif np.absolute(fp_num) > range_max:
                binary_rep[1:3] = np.fromiter(np.binary_repr(4), int)
        return binary_rep




    def cartpole_binarizer(self, input_state, n_bins=15, bin_type="S"):
        if bin_type == "B":
            # binned binarizer
            op_1 = self._binned_binarizer(input_state[0], 0, 3, n_bins-1)
            op_2 = self._binned_binarizer(input_state[1], 0, 500, n_bins-1)
            op_3 = self._binned_binarizer(input_state[2], 0, 42, n_bins-1)
            op_4 = self._binned_binarizer(input_state[3], 0, 500, n_bins-1)
        elif bin_type == "G":
            # greater_than binarizer:
            op_1 = self._greater_than_binarizer(input_state[0], n_places=n_bins-1)
            op_2 = self._greater_than_binarizer(input_state[1], n_places=n_bins-1)
            op_3 = self._greater_than_binarizer(input_state[2], n_places=n_bins-1)
            op_4 = self._greater_than_binarizer(input_state[3], n_places=n_bins-1)
        elif bin_type == "Q":
            # quartile binarizer
            op_1 = self._quartile_binner(input_state[0], 3)
            op_2 = self._quartile_binner(input_state[1], 500)
            op_3 = self._quartile_binner(input_state[2], 42)
            op_4 = self._quartile_binner(input_state[3], 500)
        elif bin_type == "U":
            # unsigned binarizer
            op_1 = self._unsigned_binarizer(input_state[0], -1.5, 1.5, n_bins)
            op_2 = self._unsigned_binarizer(input_state[1], -3.125, 3.125, n_bins)
            op_3 = self._unsigned_binarizer(input_state[2], -15, 15, n_bins)
            op_4 = self._unsigned_binarizer(input_state[3], -3.125, 3.125, n_bins)

        elif bin_type == "L":
            # lesser than binned binarizer
            # based on arXiv:1905.04199v2 [cs.LG]
            op_1 = self._less_than_binned_binarizer(input_state[0], -1.5, 1.5, n_bins)
            op_2 = self._less_than_binned_binarizer(input_state[1], -100, 100, n_bins)
            op_3 = self._less_than_binned_binarizer(input_state[2], -15, 15, n_bins)
            op_4 = self._less_than_binned_binarizer(input_state[3], -100, 100, n_bins)

        else:
            op_1 = self._simple_binarizer(input_state[0], bits=n_bins-1)
            op_2 = self._simple_binarizer(input_state[1], bits=n_bins-1)
            op_3 = self._simple_binarizer(input_state[2], bits=n_bins-1)
            op_4 = self._simple_binarizer(input_state[3], bits=n_bins-1)
        return [op_1, op_2, op_3, op_4]

    def plot_bin_dist(self, plot_file, binarizer):

        fig, axs = plt.subplots(2, 2)
        df = pd.DataFrame(columns=['cart_position', 'cart_velocity', 'pole_angle', 'pole_velocity'])
    
        for i in range(0, len(self.bin_labels), 4):
            df = df.append({'cart_position': self.bin_labels[i], 
            'cart_velocity': self.bin_labels[i+1], 
            'pole_angle':self.bin_labels[i+2], 
            'pole_velocity':self.bin_labels[i+3]}, ignore_index=True)

        df['cart_position'].value_counts().sort_index(ascending=True).plot(kind="bar", ax=axs[0][0])
        axs[0, 0].set_title('cart_position')
        df['cart_velocity'].value_counts().sort_index(ascending=True).plot(kind="bar", ax=axs[0][1])
        axs[0, 1].set_title('cart_velocity')
        df['pole_angle'].value_counts().sort_index(ascending=True).plot(kind="bar", ax=axs[1][0])
        axs[1, 0].set_title('pole_angle')
        df['pole_velocity'].value_counts().sort_index(ascending=True).plot(kind="bar", ax=axs[1][1])
        axs[1, 1].set_title('pole_velocity')

        for ax in axs.flat:
            ax.set(xlabel='bin', ylabel='frequency')

        fig.tight_layout()
        fig.suptitle(binarizer)
        plt.savefig(plot_file)

def test():
    env = gym.make("CartPole-v0")
    state = env.reset()
    print("Original State: {}".format(state))
    disc = CustomDiscretizer()
    disc_state = disc.cartpole_binarizer(state, n_bins=4, bin_type="L")
    print("Discretized State: {}".format(disc_state))
    print(disc.bin_labels)
    print("END")


if __name__ == "__main__":
    test()
