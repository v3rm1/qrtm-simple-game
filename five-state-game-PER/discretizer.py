import numpy as np
# import gym


class CustomDiscretizer:
    def __init__(self):
        super().__init__()


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
            op_1 = self._binned_binarizer(input_state, 0, 4, n_bins)
        elif bin_type == "G":
            # greater_than binarizer:
            op_1 = self._greater_than_binarizer(input_state, n_places=n_bins)
        elif bin_type == "Q":
            # quartile binarizer
            op_1 = self._quartile_binner(input_state, 4)
        elif bin_type == "U":
            # unsigned binarizer
            op_1 = self._unsigned_binarizer(input_state, 0, 4, n_bins+1)
        else:
            op_1 = self._simple_binarizer(input_state, bits=n_bins)
        return [op_1]


# def test():
#     env = gym.make("CartPole-v0")
#     state = env.reset()
#     print("Original State: {}".format(state))
#     disc = CustomDiscretizer()
#     disc_state = disc.cartpole_binarizer(state, n_bins=4)
#     print("Discretized State: {}".format(disc_state))
#     print("END")


# if __name__ == "__main__":
#     test()
