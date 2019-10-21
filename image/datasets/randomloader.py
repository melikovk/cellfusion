from torch.utils.data import Dataset, DataLoader, ConcatDataset
from . import MultiFileCropDataset

class RandomLoader(DataLoader):
    """ This is a dataloader that randomizes datasets before each iteration.
    Datasets should have *reset* method defined. This is usefull for different
    *CropDataset* that should be randomized before iteration and not during sampling.
    Dataloader redefines length so it can be used with *MultiFileCropDataset*
    derived from IterableDatatset
    """
    def __iter__(self):
        if isinstance(self.dataset, ConcatDataset):
            for dset in self.dataset.datasets:
                dset.reset()
        else:
            self.dataset.reset()
        return super().__iter__()

    def __len__(self):
        if isinstance(self.dataset, MultiFileCropDataset):
            return len(self.dataset) // (self.batch_size)
        else:
            return super().__len__()
