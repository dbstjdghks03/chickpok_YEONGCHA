from preprocessing import PreProcess
import numpy as np

import librosa
import matplotlib.pyplot as plt



class noise(object):
    def __init__(self, noise_factor) -> None:
        self.noise_factor = noise_factor

    def __call__(self, data):
        """
        원본 데이터에 노이즈를 추가합니다.
        noise factor를 통해 조절합니다.
        """
        noise = np.random.randn(len(data))
        augmented_data = data + self.noise_factor * noise
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data


class shifting(object):
    def __init__(self, sampling_rate, shift_max, shift_direction) -> None:
        self.sampling_rate = sampling_rate
        self.shift_max = shift_max
        self.shift_direction = shift_direction

    def __call__(self, data):
        """
        원본 데이터를 좌우로 이동시킵니다.
        shift_max를 통해 최대 얼마까지 이동시킬지 조절합니다.
        """
        shift = np.random.randint(self.sampling_rate * self.shift_max+1)
        if self.shift_direction == 'right':
            shift = -shift
        elif self.shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data


class change_pitch(object):
    def __init__(self, sampling_rate, pitch_factor) -> None:
        self.sampling_rate = sampling_rate
        self.pitch_factor = pitch_factor

    def __call__(self, data):
        """
        원본 데이터의 피치를 조절합니다.
        """
        return librosa.effects.pitch_shift(data, self.sampling_rate, self.pitch_factor)

