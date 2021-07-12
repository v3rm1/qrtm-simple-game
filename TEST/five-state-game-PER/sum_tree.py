import numpy as np

class SumTree:
    write_idx = 0

    def __init__(self, tree_memory_length):
        self.tree_mem_len = tree_memory_length
        self.tree = np.zeros(2 * tree_memory_length - 1)
        self.data = np.zeros(tree_memory_length, dtype=object)
        self.num_entries = 0

    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
        
    def _retrieve(self, idx, sample):
        left_idx = 2 * idx +1
        right_idx = left_idx + 1
        if left_idx >= len(self.tree):
            return idx
        
        if sample <= self.tree[left_idx]:
            return self._retrieve(left_idx, sample)
        else:
            return self._retrieve(right_idx, sample - self.tree[left_idx])
        
    def sum_of_tree(self):
        return self.tree[0]
    
    def add(self, priority, data):
        idx = self.write_idx + self.tree_mem_len - 1
        self.data[self.write_idx] = data
        self.update_priority(idx, priority)
        
        self.write_idx += 1
        if self.write_idx >= self.tree_mem_len:
            self.write_idx = 0
        
        if self.num_entries < self.tree_mem_len:
            self.num_entries += 1
    
    def update_priority(self, idx, priority):
        delta_priority = priority - self.tree[idx]

        self.tree[idx] = priority
        self._propagate(idx, delta_priority)

    def get_sample(self, sample):
        idx = self._retrieve(0, sample)
        data_idx = idx - self.tree_mem_len + 1
        return (idx, self.tree[idx], self.data[data_idx])

