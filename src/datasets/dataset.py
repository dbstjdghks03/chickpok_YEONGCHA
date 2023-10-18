from torch.utils.data import Dataset, DataLoader
import os
from sklearn import preprocessing
import json
import re
from nptdms import TdmsFile
import numpy as np
import torchaudio
import torch
import librosa

__all__ = ['YoungDataLoader', 'TrainDataSet', 'FERTestDataSet']


def get_numpy_from_nonfixed_2d_array(input, fixed_length=4000000, padding_value=0):
    output = np.pad(input, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length]
    return output


train_to_idx = {
    '수소열차': 0,
    '차세대전동차': 1
}
class_to_idx = {
    'Yes': 0,
    'No': 1
}


def tdms_preprocess(tdms_path):
    tdms_file = TdmsFile(tdms_path)
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

    y = (data_sum / peakAmp) * maxPeak
    Beampower = None
    return y, Beampower


class PreProcess:
    def __init__(self, tdms_file):
        # print(tdms_file['RawData'], tdms_file['RawData'].channels())
        # L = list(name for name in tdms_file['RawData'].channels())
        # L_str = list(map(str, L))
        # data_lst = []
        # peak_lst = []
        # for string in L_str:
        #     num = re.sub(r'[^0-9]', '', string)
        #     if num:
        #         selected_data = tdms_file['RawData'][f'Channel{num}']
        #         data_lst.append(selected_data.data)
        #         peak_lst.append(max(abs(selected_data.data)))
        # data_sum = sum(data_lst)
        # peakAmp = max(abs(data_sum))
        # maxPeak = max(peak_lst)
        #
        # self.y = get_numpy_from_nonfixed_2d_array((data_sum / peakAmp) * maxPeak)
        self.y = tdms_file

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
        arr = np.array(flat).reshape(3, amplitude.shape[0], amplitude.shape[1])

        return arr

    def get_mfcc(self):
        # mfcc = librosa.feature.mfcc(y=self.y, sr=22050, n_mfcc=10, n_fft=640, hop_length=256)
        # mfcc = preprocessing.scale(mfcc, axis=1)

        y = torch.tensor(self.y)  # If `self.y` is not already a tensor

        # Compute MFCC using torchaudio
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=22050,
            n_mfcc=10,
            melkwargs={
                "n_fft": 640,
                "hop_length": 256,
                "center": True,  # default behavior in librosa, adjust if needed
            }
        )(y)

        # Scaling the MFCC (equivalent to preprocessing.scale in sklearn)
        mean = torch.mean(mfcc, dim=1, keepdim=True)
        std = torch.std(mfcc, dim=1, keepdim=True)
        mfcc = (mfcc - mean) / std

        return self.getrgb(mfcc, mfcc.min(), mfcc.max())

    def get_stft(self):
        x = self.y

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        specgram = torchaudio.transforms.Spectrogram(
            n_fft=640,
            win_length=640,
            hop_length=256,
            power=None  # To get complex output, not magnitude squared
        )(x)

        # Compute magnitude
        magnitude = specgram.abs()

        # Convert to dB scale
        log_spectrogram = torchaudio.transforms.AmplitudeToDB()(magnitude)
        return self.getrgb(log_spectrogram, log_spectrogram.min(), log_spectrogram.max())

    def get_sc(self):
        # cent = librosa.feature.spectral_centroid(y=self.y, sr=22050).reshape(-1, 1)

        y = torch.tensor(self.y)  # If `self.y` is not already a tensor
        sample_rate = 22050

        # Compute the Spectrogram
        specgram = torchaudio.transforms.Spectrogram(n_fft=640, win_length=640, hop_length=256, power=None )(y)

        # Compute Spectral Centroid
        frequencies = torch.linspace(0, sample_rate / 2, specgram.shape[1])
        centroid = torch.sum(frequencies * specgram) / torch.sum(specgram)

        print(centroid)
        # Reshape the tensor
        cent = centroid.reshape(-1, 1)
        return cent


class YoungDataSet(Dataset):
    def __init__(self, root, is_npy, transform=None):
        self.transform = transform
        self.data_list = []
        self.root = root
        print(root + '/train_json')
        for dirpath, dirnames, files in os.walk(root + '/train_json'):
            print(f'Found directory: {dirpath}')
            for file_name in files:
                if file_name.endswith(".json"):
                    with open(dirpath + '/' + file_name, 'r') as f:
                        data = json.load(f)
                    folder_name = dirpath.split("/")[-1]
                    s206_path = os.path.join(root, '/train_tdms', folder_name, 'S206', data['title_s206'])
                    if is_npy:
                        s206_path = os.path.join(root, '/train_tdms', folder_name, 'S206',
                                                 data['title_s206'].split(".")[0] + ".npy")

                    batcam_path = os.path.join(root, '/train_tdms', folder_name, 'BATCAM2',
                                               data['title_batcam2'])
                    train = train_to_idx[data['Train']]
                    horn = class_to_idx[data['Horn']]

                    if horn == 0:
                        position = int(data['Position'])
                        if train == 0:
                            cluster = 'CL_HY'
                        else:
                            cluster = 'CL_NY'
                    else:
                        position = -1
                        if train == 0:
                            cluster = 'CL_HN'
                        else:
                            cluster = 'CL_NN'

                    self.data_list.append([s206_path, batcam_path, train, horn, position, cluster, data])

    def __getitem__(self, idx):
        s206_path, batcam_path, _, horn, position, _, _ = self.data_list[idx]
        s206_audio = np.load(self.root+s206_path)
        # s206_audio = TdmsFile(self.root + s206_path)
        # batcam_audio, batcam_beam = tdms_preprocess(self.root + batcam_path)
        print(self.root + s206_path, s206_audio)
        s206 = PreProcess(s206_audio)
        print(s206.get_stft().shape, s206.get_mfcc().shape, s206.get_sc().shape, horn)

        return torch.tensor(s206.get_stft()), torch.tensor(s206.get_mfcc()), torch.tensor(
            s206.get_sc()), horn, torch.tensor(position)
        # if self.transform:
        #     self.data[index] = AudioAugs(self.transform, sampling_rate, p=0.5)

        # return s206_audio, batcam_audio, s206_beam, batcam_beam, label[0], label[1], data

        # return self.data[index], self.emotion[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    data = YoungDataSet(root="/home/gritte/workspace/MyCanvas/data/data_set")
    print(data)
