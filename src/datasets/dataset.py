import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import pandas as pd
import os
# import numpy as np
import re
import numpy as np
import pandas as pd
# import librosa
from sklearn import preprocessing
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.cm as cm
import json
# from src.data.preprocessing import PreProcess
# # from src.data.preprocessing import AudioAugs

__all__ = ['YoungDataLoader', 'TrainDataSet', 'FERTestDataSet']


class YoungDataLoader(DataLoader):
    def __init__(self, batch_size, root, shuffle=True, num_workers=0):
        trsfm = None
        trsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-45, 45)),
            transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor()
        ])

        self.dataset = YoungDataSet(transform=trsfm, root=root)
        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class YoungDataSet(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data_list = []
        print(root + '/train_json')
        for dirpath, dirnames, files in os.walk(root + '/train_json'):
            print(f'Found directory: {dirpath}')
            for file_name in files:
                if file_name.endswith(".json"):
                    with open(dirpath + '/' + file_name, 'r') as f:
                        data = json.load(f)
                    folder_name = dirpath.split("/")[-1]
                    s206_path = os.path.join(root, '/train_tdms', folder_name, 'S206', data['title_s206'])
                    batcam_path = os.path.join(root, '/train_tdms', folder_name, 'BATCAM2',
                                               data['title_batcam2'])

                    label = (data['Horn'], data['Position'])
                    self.data_list.append((s206_path, batcam_path, label, data))
        print(self.data_list)
        self.len = len(self.data_list)

    def __getitem__(self, idx):
        s206_path, batcam_path, label, data = self.data_list[idx]

        s206_audio, s206_beam = self.tdms_preprocess(s206_path)
        batcam_audio, batcam_beam = self.tdms_preprocess(batcam_path)

        s206 = PreProcess(s206_audio)
        s206.get_stft()

        return s206.get_stft(), s206.get_mfcc(), s206.get_sc(), label[0]
        # if self.transform:
        #     self.data[index] = AudioAugs(self.transform, sampling_rate, p=0.5)

        # return s206_audio, batcam_audio, s206_beam, batcam_beam, label[0], label[1], data

        # return self.data[index], self.emotion[index]

    def __len__(self):
        return self.len


class FERTestDataLoader(DataLoader):
    def __init__(self, batch_size=1, shuffle=False, num_workers=0):
        trsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        self.dataset = FERTestDataSet(transform=trsfm)
        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class FERTestDataSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        dir_path = os.path.dirname(os.path.realpath(__file__))

        if os.path.isfile(dir_path + '/cache/test/data.pt') and os.path.isfile(dir_path + '/cache/test/label.pt'):
            self.emotion = torch.load(dir_path + '/cache/test/label.pt')
            self.data = torch.load(dir_path + '/cache/test/data.pt')
            self.len = self.emotion.shape[0]
            return

        df = pd.read_csv('dataset/fer2013.csv')
        df = df[(df['Usage'] == 'PrivateTest') | (df['Usage'] == 'PublicTest')]
        self.emotion = torch.LongTensor(df['emotion'].values)
        self.data = df['pixels'].apply(
            lambda a: torch.FloatTensor(list(map(int, a.split(' ')))).reshape(1, 48, 48)).values
        self.len = self.emotion.shape[0]

        torch.save(self.data, 'data_loader/cache/test/data.pt')
        torch.save(self.emotion, 'data_loader/cache/test/label.pt')

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index]), self.emotion[index]
        return self.data[index], self.emotion[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    data = YoungDataSet(root="/home/gritte/workspace/MyCanvas/data/data_set")
    print(data)

