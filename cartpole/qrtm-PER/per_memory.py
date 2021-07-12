import random
import numpy as np
from sum_tree import SumTree


class Memory:
    def __init__(self, tree_memory_length, error_multiplier=0.01, alpha=0.6, beta=0.4, beta_increment_per_sample=0.001):
        self.tree = SumTree(tree_memory_length)
        self.tree_memory_length = tree_memory_length
        self.error_multiplier = error_multiplier
        self.per_alpha = alpha
        self.per_beta_init = beta
        self.beta_increment_per_sample = beta_increment_per_sample

    def _get_priority(self, error):
        return (np.abs(error) + self.error_multiplier) ** self.per_alpha
    
    def add_sample_to_tree(self, error, sample):
        priority = self._get_priority(error)
        self.tree.add(priority, sample)

    def sample_tree(self, num_samples):
        batch = []
        idxs = []
        segment = self.tree.sum_of_tree() / num_samples
        priorities = []

        self.beta = np.min([1.0, self.per_beta_init + self.beta_increment_per_sample])

        for i in range(num_samples):
            a = segment * i
            b = segment * (i + 1)

            sample = random.uniform(a, b)
            idx, priority, data = self.tree.get_sample(sample)

            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)
        
        sampling_prob = priorities / self.tree.sum_of_tree()
        is_weight = np.power(self.tree.num_entries * sampling_prob, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight
    
    def update_tree(self, idx, error):
        priority = self._get_priority(error)
        self.tree.update_priority(idx, priority)
