from torch.utils.data import Sampler
import itertools
import numpy as np

class DynamicBatchSampler(Sampler):
    def __init__(self, data_source, batch_sizes):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_sizes = itertools.cycle(batch_sizes)
        assert len(batch_sizes) == 1 or np.sum(batch_sizes) == len(data_source), \
            f'Invalid batch_sizes: {batch_sizes}, data_source: {len(data_source)}'
        self.batch_indices = self._create_batch_indices()
        assert len(batch_sizes) == 1 or len(self.batch_indices) == len(batch_sizes), \
            f'Invalid batch_indices: {len(self.batch_indices)}, batch_sizes: {len(batch_sizes)}'

    def _create_batch_indices(self):
        indices = list(range(len(self.data_source)))
        batch_indices = []
        batch_size = next(self.batch_sizes)
        while indices:
            if len(indices) < batch_size:
                batch_size = len(indices)
            batch_indices.append(indices[:batch_size])
            indices = indices[batch_size:]
            batch_size = next(self.batch_sizes)
        return batch_indices

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)