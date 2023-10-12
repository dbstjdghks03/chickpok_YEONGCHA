from preprocessing import PreProcess
import torch
import numpy as np
import random
import librosa
import matplotlib.pyplot as plt



class data_aug:
    def __init__(self, data) -> None:
        self.data = data

    '''def comparison(func, y):
        aug_y = func(y, 0.1)

        mfcc = librosa.feature.mfcc(y=y, sr=25600, n_mfcc=100, n_fft=640, hop_length=256)
        aug_mfcc = librosa.feature.mfcc(y=aug_y, sr=25600, n_mfcc=100, n_fft=640, hop_length=256)

        #librosa.display.specshow(mfcc, sr=25600, x_axis='ms')
        librosa.display.specshow(aug_mfcc, sr=25600, x_axis='ms')'''


    def noising(self, noise_factor):
        """
        원본 데이터에 노이즈를 추가합니다.
        noise factor를 통해 조절합니다.
        """
        noise = np.random.randn(len(self.data))
        augmented_data = self.data + noise_factor * noise
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(self.data[0]))
        return augmented_data

    def shifting(self, sampling_rate, shift_max, shift_direction):
        """
        원본 데이터를 좌우로 이동시킵니다.
        shift_max를 통해 최대 얼마까지 이동시킬지 조절합니다.
        """
        shift = np.random.randint(sampling_rate * shift_max+1)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(self.data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data

    def change_pitch(self, sampling_rate, pitch_factor):
        """
        원본 데이터의 피치를 조절합니다.
        """
        return librosa.effects.pitch_shift(self.data, sampling_rate, pitch_factor)

