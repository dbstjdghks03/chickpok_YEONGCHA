"""
This directory holds scripts that handle data processing.

It may include files like
"make_dataset.py" (a script to download, filter, preprocess, and partition the raw data into training and test splits),
"preprocessing.py" (a script containing functions for data cleaning and preparation for modeling).
"""
import torch
from torch.utils.data import Dataset

class AudioFeatureDataset(Dataset):
    def __init__(self, stft_data, mfcc_data, sc_data, labels):
        self.stft_data = stft_data
        self.mfcc_data = mfcc_data
        self.sc_data = sc_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        stft = self.stft_data[idx]
        mfcc = self.mfcc_data[idx]
        sc = self.sc_data[idx]
        label = self.labels[idx]

        return stft, mfcc, sc, label

#데이터 로딩 및 DataLoader 생성:
# 예를 들어, stft_data, mfcc_data, sc_data, labels를 준비한 후
Data = AudioFeatureDataset()

custom_dataset = AudioFeatureDataset(Data.stft, Data.mfcc, Data.sc, Data.label)

# DataLoader 생성
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=32, shuffle=True)