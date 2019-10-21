from torch.utils.data import IterableDataset, get_worker_info
from itertools import chain
import random

class MultiFileCropDataset(IterableDataset):
    def __init__(self, fileslist, dataset_class, dataset_params):
        super().__init__()
        self.fileslist = fileslist
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            fileslist = self.fileslist
        else:
            fnum = len(self.fileslist) // worker_info.num_workers
            id = worker_info.id
            fileslist = self.fileslist[fnum*id:fnum*(id+1)]
        return chain.from_iterable((self.dataset_class(*names, **self.dataset_params) for names in fileslist))

    def __len__(self):
        """This is a temporary hack"""
        return len(self.fileslist)*self.dataset_params['length']

    def reset(self):
        random.shuffle(self.fileslist)
