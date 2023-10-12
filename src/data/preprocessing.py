import re
from nptdms import TdmsFile
import numpy as np
import torchaudio
import torch
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

        self.y = torch.tensor((data_sum / peakAmp) * maxPeak)

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
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=25600,
            n_mfcc=100,
            melkwargs={"n_fft": 640, "hop_length": 256, "n_mels": 23}
            # default n_mels=23, you can adjust based on your requirements
        )

        mfcc = mfcc_transform(self.y)

        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True)
        mfcc = (mfcc - mean) / std
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
        x = self.y

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        # Compute STFT
        specgram = torchaudio.transforms.Spectrogram(
            n_fft=640,
            win_length=640,
            hop_length=256,
            power=None  # To get complex output, not magnitude squared
        )(x)

        # Calculate magnitude
        magnitude = specgram.abs()

        # Compute the spectral centroid
        freqs = torch.linspace(0, 25600 / 2, magnitude.size(1), dtype=torch.float)
        spectral_centroid = torch.sum(freqs * magnitude) / torch.sum(magnitude)

        '''times = librosa.times_like(cent)
        # librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), y_axis='linear', x_axis='time')
        plt.plot(times, cent.T, color='black')
        plt.axis('off')
        plt.show()
        plt.savefig('sc.jpg')
        sc_jpg = Image.open('stft.jpg')
        sc_arr = np.array(sc_jpg)'''

        return spectral_centroid
