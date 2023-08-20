import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import minmax_scale
import pandas as pd


class MTSDataset(Dataset):
    def __init__(self, raw_seqs, win_len=20, selected_features=None, minmax=True) -> None:
        super().__init__()

        if minmax:
            raw_seqs = minmax_scale(raw_seqs)

        self._raw_seqs = torch.tensor(raw_seqs, dtype=torch.float32)
        self._win_len = win_len

    def __len__(self):
        return len(self._raw_seqs) - self._win_len + 1

    
    def __getitem__(self, index):
        return self._raw_seqs[index : index + self._win_len]




class UTSDataset(Dataset):
    def __init__(self, raw_seqs, win_len=20, minmax=True) -> None:
        super().__init__()

        if minmax:
            raw_seqs = minmax_scale(raw_seqs)

        self._raw_seqs = torch.tensor(raw_seqs, dtype=torch.float32)
        self._win_len = win_len

    def __len__(self):
        return len(self._raw_seqs) - self._win_len + 1

    
    def __getitem__(self, index):
        return self._raw_seqs[index : index + self._win_len]

    def set_win_len(self, win_len):
        self._win_len = win_len


class UTSTestDataset(Dataset):
    def __init__(self, raw_seqs, labels, win_len=20, minmax=True) -> None:
        super().__init__()

        if minmax:
            raw_seqs = minmax_scale(raw_seqs)

        self._raw_seqs = torch.tensor(raw_seqs, dtype=torch.float32)
        self._labels = torch.tensor(labels, dtype=torch.bool)
        self._win_len = win_len

    def __len__(self):
        return len(self._raw_seqs) - self._win_len + 1

    
    def __getitem__(self, index):
        return self._raw_seqs[index : index + self._win_len], self._labels[index + self._win_len - 1]


    def set_win_len(self, win_len):
        self._win_len = win_len



class TZSDataset(UTSDataset):
    def __init__(self, kpi_path, win_len=20, minmax=True) -> None:
        seq = pd.read_csv(kpi_path)['value'].to_numpy()
        super().__init__(seq, win_len, minmax)



class TZSTestDataset(UTSTestDataset):
    def __init__(self, kpi_path, win_len=20, minmax=True) -> None:
        seq = pd.read_csv(kpi_path)[['value', 'label']].to_numpy()
        raw_seq = seq[:, 0]
        labels = seq[:, 1]
        
        super().__init__(raw_seq, labels, win_len, minmax)