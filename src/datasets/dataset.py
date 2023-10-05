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
from src.data.preprocessing import tdms_preprocess
# # from src.data.preprocessing import AudioAugs

__all__ = ['TrainDataLoader', 'TrainDataSet', 'FERTestDataSet']


class TrainDataLoader(DataLoader):
    def __init__(self, batch_size, root, shuffle=True, num_workers=0):
        trsfm = None
        # trsfm = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation((-45, 45)),
        #     transforms.ColorJitter(brightness=0.5),
        #     transforms.ToTensor()
        # ])

        self.dataset = TrainDataSet(transform=trsfm, root=root)
        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class TrainDataSet(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.meta = []
        self.len = 100
        for dirpath, dirnames, files in os.walk(root+'/data/raw_data/train_json'):
            print(f'Found directory: {dirpath}')
            for file_name in files:
                if file_name.endswith(".json"):
                    with open(dirpath+'/'+file_name, 'r') as f:
                        data = json.load(f)
                    folder_name = dirpath.split("/")[-1]
                    s206_audio, s206_beam = tdms_preprocess(root+ '/data/raw_data/train_tdms/' + folder_name+'/S206/'+data['title_s206'])
                    batcam_audio, batcam_beam = tdms_preprocess(root+ '/data/raw_data/train_tdms/' + folder_name+'/BATCAM2/'+data['title_batcam2'])

                    print(s206_audio, batcam_audio, data)

    def __getitem__(self, index):
        if self.transform:
            self.data[index] = AudioAugs(self.transform, sampling_rate, p=0.5)
        return self.data[index], self.emotion[index]

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


class PreProcess:
    def __init__(self, tdms_file):
        L = list(name for name in tdms_file['RawData'].channels())
        L_str = list(map(str, L))
        data_lst = []
        peak_lst = []
        for string in L_str:
            num = re.sub(r'[^0-9]', '', string)
            if num:
                selected_data = tdms_file['RawData'][f'Channel{num}']
                data_lst.append(selected_data.data)
                peak_lst.append(max(abs(selected_data.data)))
        data_sum = sum(data_lst)
        peakAmp = max(abs(data_sum))
        maxPeak = max(peak_lst)

        self.y = (data_sum / peakAmp) * maxPeak

    def getrgb(self, amplitude, min_amplitude=0, max_amplitude=10):
        # 진폭값을 [0, 1] 범위로 정규화
        normalized_amplitude = (amplitude - min_amplitude) / (max_amplitude - min_amplitude)

        flat = list(normalized_amplitude.flatten())

        for i in range(len(flat)):
          # RGB 색상 계산
          r = int(flat[i] * 255)
          g = 0  # 여기서는 녹색 채널을 0으로 설정
          b = 255 - r  # 파란색 채널을 반전
          flat[i] = [r, g, b]
        arr = np.array(flat).reshape(amplitude.shape[0], amplitude.shape[1], 3)

        return arr

    def get_mfcc(self):
        mfcc = librosa.feature.mfcc(y=self.y, sr=25600, n_mfcc=100, n_fft=640, hop_length=256)
        mfcc = preprocessing.scale(mfcc, axis=1)
        '''pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, 6700)'''

        '''# 색상 매핑 (RGB)
        cmap = cm.get_cmap('plasma')  # Here 'viridis' is the color map. You can use others like 'plasma', 'inferno', etc.
        rgb_image = cmap(padded_mfcc)  # This will be a 3D array with dimensions: [height, width, 3 (for RGB channels)]
        print(rgb_image.shape)
        plt.figure(figsize=(10, 5))
        plt.imshow(rgb_image)
        plt.axis('off')  # 축을 숨기려면 이 줄을 추가
        plt.savefig('mfcc.jpg')
        plt.show()'''

        # return self.getrgb(padded_mfcc, padded_mfcc.min(), padded_mfcc.max())
        return self.getrgb(mfcc, mfcc.min(), mfcc.max())

    def get_stft(self):
        x = self.y
        stft = librosa.stft(x, n_fft=640, hop_length=256, win_length=640)

        magnitude = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)

        return self.getrgb(log_spectrogram, log_spectrogram.min(), log_spectrogram.max())

    def get_sc(self):
        cent = librosa.feature.spectral_centroid(y=self.y, sr=25600, n_fft=640, win_length=640, hop_length=256)

        '''times = librosa.times_like(cent)
        # librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), y_axis='linear', x_axis='time')
        plt.plot(times, cent.T, color='black')
        plt.axis('off')
        plt.show()
        plt.savefig('sc.jpg')
        sc_jpg = Image.open('stft.jpg')
        sc_arr = np.array(sc_jpg)'''

        return cent