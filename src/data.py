import cupy as cp
from src.tensor import Tensor

class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
    
class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]
    
class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        self.indices = cp.arange(len(self.dataset))
        if self.shuffle:
            cp.random.shuffle(self.indices)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.idx:self.idx + self.batch_size]

        if self.drop_last and len(batch_indices) < self.batch_size:
            raise StopIteration

        batch = [self.dataset[i] for i in batch_indices]
        self.idx += self.batch_size

        if isinstance(batch[0], tuple):
            return tuple(Tensor(cp.stack([item.data for item in items])) for items in zip(*batch))
        else:
            return Tensor(cp.stack([item.data for item in batch]))