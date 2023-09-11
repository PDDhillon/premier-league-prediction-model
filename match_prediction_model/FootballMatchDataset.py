from torch.utils.data import Dataset
import torch

class FootballMatchDataset(Dataset):
    def __init__(self, data_df):
        super().__init__()
        self.data = data_df

    def __getitem__(self, index):
        record = self.data.iloc[index]
        features = record[:6]
        label = record[-1]
        return (torch.tensor(features.values), label)
    
    def __len__(self):
        return len(self.data)