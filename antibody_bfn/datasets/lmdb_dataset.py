import lmdb
import pickle
import torch
from torch.utils.data import Dataset
from ._base import register_dataset

class LMDBDataset(Dataset):
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self.env = None
        self.keys = None
        self.length = 0
        
        # Open to check length
        env = lmdb.open(db_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            self.length = txn.stat()['entries']
        env.close()

    def _connect(self):
        if self.env is None:
            self.env = lmdb.open(
                self.db_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._connect()
        # Keys are stored as 8-digit integers
        key = f"{index:08d}".encode()
        with self.env.begin() as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                # Should not happen if length is correct, but safe fallback
                return self.__getitem__((index + 1) % self.length)
            return pickle.loads(data_bytes)

@register_dataset('lmdb_preprocessed')
def get_lmdb_dataset(cfg, transform=None):
    # Note: transform is ignored because data is already transformed!
    return LMDBDataset(db_path=cfg.db_path)
